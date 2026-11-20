"""Sample i.i.d. schedules from a trained UGNN: p(schedule | conflict graph).

GPU: one process loops over the graphs, a graph's T draws tiled into one block-diagonal batch.
CPU: the common.bench spawn pool, one core per graph, one draw at a time (the timing convention
shared with SAGNN / MWIS).

    python -m ugnn.generate --ckpt <path or W&B run name> --data <graphs.pt> --T 100 [--device cpu] \
        [--artifact-name test_schedules_<arm> [--run-name R] [--meta k=v ...]]   upload the schedules to W&B
        [--keep-local PATH]                                                      keep the payload on disk
        [--timing-out timing.json [--arm LABEL]]                                 write the common.bench timing block
        [--delta 0.1]        min-rate target to condition on (delta-conditioned checkpoints)

Payload: {"schedules": [(T, K) uint8 per graph], "gen_seconds": [per graph], "meta": {...}}; the
notebooks pull it with wandb.Api().artifact("<project>/<artifact-name>:latest").
"""
import argparse
import json
import os
import shutil
import tempfile
import time
from types import SimpleNamespace

import torch
import wandb
from torch_geometric.data import Batch, Data
from tqdm import tqdm

import common.utils  # noqa: F401  registers the 'utils'/'SAWLData' alias for graph pickles
from common.bench import run_pool, summarize_times, timing_meta
from graph_signal_diffusion.diffusion import ddim, ddpm
from ugnn.dataset import DEFAULT_SPLIT, delta_cond, graph_A, load_samples
from ugnn.model import PROJECT, load_trained

ddim.tqdm = ddpm.tqdm = SimpleNamespace(tqdm=lambda it, **_: it)   # no per-step sampler progress bars


@torch.inference_mode()
def generate_schedules(diffusion, A, device, T, draw_batch, delta=None, bf16=False):
    """T binary (T, K) schedules for one graph, ``draw_batch`` draws per sample call -> (schedules, seconds).

    ``bf16`` runs the sampler under bfloat16 autocast (weights stay fp32); the schedule is the sign of x_0, so it
    is insensitive to the lost precision.

    ``delta`` is the min-rate target the schedules are drawn for; required iff the checkpoint was
    trained on samples carrying one (see ugnn.dataset.delta_cond).  A checkpoint trained with
    ``--window W`` emits W consecutive slots per draw, so only ceil(T/W) draws are needed and the
    returned trajectory is trimmed back to exactly T rows.
    """
    conditioned = diffusion.model.config.has_conditioning
    if conditioned and delta is None:
        raise ValueError("this checkpoint is delta-conditioned: pass a delta "
                         "(--delta, or use samples whose graphs carry one)")
    if delta is not None and not conditioned:
        raise ValueError("this checkpoint is graph-only, so --delta would be silently ignored")
    # a windowed checkpoint emits W CONSECUTIVE slots per draw (window sits on the signal
    # channels, see ugnn.dataset.window_target), so one sample call yields W rows of the output.
    W = int(diffusion.model.config.out_channels)
    K = A.shape[0]
    graph = Data(edge_index=A.nonzero().t().contiguous().to(device), num_nodes=K,
                 x=delta_cond(delta, K, device))
    out, drawn, t0 = [], 0, time.perf_counter()
    while drawn < T:
        n = max(1, min(draw_batch, -(-(T - drawn) // W)))         # blocks still needed
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            x0 = diffusion.sample((n, 1, K, W), device, Batch.from_data_list([graph] * n))
        block = x0.reshape(n, K, W).permute(0, 2, 1).reshape(n * W, K)   # (n*W, K), slot order kept
        out.append((block >= 0).float().cpu())                    # x_0 >= 0  <=>  raw schedule >= 0.5
        drawn += n * W
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(out)[:T], time.perf_counter() - t0


_W = {}                                                             # per-worker state of the CPU pool


def _graph_delta(delta, g):
    """Explicit delta wins; otherwise the graph's own (samples carry one after a --delta-max run)."""
    return g.get("delta") if delta is None else delta


def _init_worker(ckpt, T, seed, delta, sampling_timesteps):
    torch.set_num_threads(1)
    _W.update(diffusion=load_trained(ckpt, "cpu", sampling_timesteps=sampling_timesteps), T=T, seed=seed, delta=delta)


def _work(task):
    idx, g = task
    torch.manual_seed(_W["seed"] * 1_000_003 + idx)
    schedules, seconds = generate_schedules(_W["diffusion"], graph_A(g), torch.device("cpu"), _W["T"],
                                            draw_batch=1, delta=_graph_delta(_W["delta"], g))
    return idx, schedules, seconds


def rollout_ugnn(ckpt, graphs, T, device="cuda", workers=1, seed=0, delta=None, sampling_timesteps=50, bf16=False):
    """T schedules per graph -> (list of (T, K) 0/1 tensors, list of seconds), aligned with ``graphs``.

    ``delta`` overrides every graph's own min-rate target; leave it None to use each graph's.
    ``sampling_timesteps`` is the number of DDIM steps; ``bf16`` (GPU only) samples under bfloat16 autocast.
    """
    if device == "cpu":
        return run_pool(graphs, init_fn=_init_worker, init_args=(ckpt, T, seed, delta, sampling_timesteps), work_fn=_work,
                        workers=min(workers, len(graphs)), desc="UGNN (cpu pool)")
    diffusion, device = load_trained(ckpt, device, sampling_timesteps=sampling_timesteps), torch.device(device)
    torch.manual_seed(seed)
    results = [generate_schedules(diffusion, graph_A(g), device, T, draw_batch=T, delta=_graph_delta(delta, g), bf16=bf16)
               for g in tqdm(graphs, desc="UGNN (gpu)")]
    return [r[0] for r in results], [r[1] for r in results]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint path, or the W&B run name")
    p.add_argument("--data", required=True, help="samples .pt (only each graph's 'A' is used)")
    p.add_argument("--split", default=DEFAULT_SPLIT, help="dict key of the graph list, e.g. 'test' for a raw dataset .pt")
    p.add_argument("--T", type=int, default=100, help="schedules per graph")
    p.add_argument("--max-graphs", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delta", type=float, default=None,
                   help="min-rate target to condition on, for a delta-conditioned checkpoint "
                        "(default: each graph's own 'delta' from the samples file)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=min(24, (os.cpu_count() or 4) - 2), help="CPU pool size")
    p.add_argument("--wandb-project", default=PROJECT)
    p.add_argument("--artifact-name", default=None, help="upload the schedules as this W&B artifact")
    p.add_argument("--run-name", default=None, help="W&B run name (default: the artifact name)")
    p.add_argument("--meta", nargs="*", default=[], help="extra key=value pairs for the artifact metadata")
    p.add_argument("--keep-local", default=None, help="keep the schedules payload at this path")
    p.add_argument("--timing-out", default=None, help="write the timing block to this .json")
    p.add_argument("--arm", default="", help="label recorded in the timing block")
    p.add_argument("--sampling-timesteps", type=int, default=50, help="DDIM steps per sample call")
    p.add_argument("--bf16", action="store_true", help="GPU: sample under bfloat16 autocast")
    args = p.parse_args()

    graphs = load_samples(args.data, args.split)[: args.max_graphs]
    schedules, seconds = rollout_ugnn(args.ckpt, graphs, args.T, args.device, args.workers, args.seed, args.delta,
                                      args.sampling_timesteps, args.bf16)
    meta = {"ckpt": args.ckpt, "data": args.data, "T": args.T, "n_graphs": len(graphs), "seed": args.seed,
            "device": args.device, "delta": args.delta, "sampling_timesteps": args.sampling_timesteps, "bf16": args.bf16,
            **dict(kv.split("=", 1) for kv in args.meta)}
    path = args.keep_local or os.path.join(tempfile.mkdtemp(), "schedules_ugnn.pt")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"schedules": [s.to(torch.uint8) for s in schedules], "gen_seconds": seconds, "meta": meta}, path)
    if args.artifact_name:
        run = wandb.init(project=args.wandb_project, name=args.run_name or args.artifact_name, job_type="rollout", config=meta)
        artifact = wandb.Artifact(args.artifact_name, type="schedules", metadata=meta)
        artifact.add_file(path, name="schedules_ugnn.pt")
        run.log_artifact(artifact)
        run.finish()
        print(f"uploaded {args.artifact_name}: {len(graphs)} graphs x {args.T} schedules ({args.device})")
    if not args.keep_local:
        shutil.rmtree(os.path.dirname(path))
    if args.timing_out:
        timing = timing_meta("UGNN", summarize_times(seconds, args.T), workers=args.workers,
                             schedules_per_graph=args.T, arm=args.arm, ckpt=args.ckpt, device=args.device)
        os.makedirs(os.path.dirname(os.path.abspath(args.timing_out)), exist_ok=True)
        with open(args.timing_out, "w") as f:
            json.dump(timing, f, indent=1)
        print(f"{args.arm}: {timing['mean_seconds_per_graph']:.3f} s/graph, {timing['mean_ms_per_schedule']:.1f} ms/schedule -> {args.timing_out}")


if __name__ == "__main__":
    main()
