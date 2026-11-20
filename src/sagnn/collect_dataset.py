from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

import common.utils  # noqa: F401  registers the 'utils' alias for old dataset pickles
from common.utils import mask_schedule_trajectory
from common.bench import run_pool, summarize_times, print_timing_summary, timing_meta
from sagnn.rollout import load_sagnn, sagnn_dual_rollout, sagnn_dual_rollout_graph
from ugnn.dataset import DEFAULT_SPLIT, graph_A


def load_graphs(path: str, split=DEFAULT_SPLIT):
    """Load the per-graph list (each dict carries at least ``A``) from a samples file."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if split not in obj:
            raise KeyError(f"split {split!r} not in {path}; keys={list(obj.keys())}")
        obj = obj[split]
    return list(obj)


def post_mask_dataset(graphs, seed: int = 0):
    """Post-hoc masking: collision-free the *saved* schedule (no re-rollout).

    For each graph, apply ``remove_collisions`` to every timestep of the existing
    (raw, unmasked) ``raw_schedule`` and store the collision-free CONTINUOUS
    schedule (``raw_schedule * survivor_mask``) as the new ``raw_schedule``; the
    survivor mask becomes ``thresholded_schedule``.  The SAGNN trajectory is
    bit-identical to the source dataset — only the saved schedule is masked.
    Each graph uses a deterministic ``np.random.Generator(seed + index)``.
    """
    out = []
    for i, g in enumerate(tqdm(graphs, desc="post-mask", unit="graph")):
        A = torch.as_tensor(g["A"], dtype=torch.float32)
        raw = torch.as_tensor(g["raw_schedule"], dtype=torch.float32)        # (T, K) raw continuous
        rng = np.random.default_rng(seed + i)
        mask = torch.as_tensor(
            mask_schedule_trajectory(raw, A, threshold=0.5, rng=rng, zero_if_all_active=True),
            dtype=torch.float32,
        )                                                                    # (T, K) 0/1 survivor; all-on -> zero
        out.append({
            "A": A,
            "raw_schedule": raw * mask,                                       # masked continuous (target)
            "thresholded_schedule": mask,                                     # collision-free survivor mask
        })
    return out


# ── parallel SAGNN rollout (one worker per graph) ─────────────────────────────
_W = {}  # per-worker state (model, params)


def _init_rollout_worker(model_path, params):
    torch.set_num_threads(1)  # one core per worker -> no oversubscription across the pool
    _W["model"] = load_sagnn(model_path, device="cpu")
    _W["params"] = params


@torch.no_grad()
def _rollout_work(task):
    idx, g = task
    model, p = _W["model"], _W["params"]
    rng = np.random.default_rng(p["seed"] + idx) if p["mask_collisions"] else None
    raw, binary, _lam, gen_seconds = sagnn_dual_rollout_graph(
        model, graph_A(g), torch.device("cpu"), T=p["T"], lr_lambda=p["lr_lambda"],   # dense A from edge_index if needed
        delta=p["delta"], weight_decay=p["weight_decay"], T0=p["T0"],
        mask_collisions=p["mask_collisions"], rng=rng, return_time=True,
    )
    w = p["warmup_steps"]
    if w:                                                       # drop cold-start transient -> T - warmup slots
        raw, binary = raw[w:], binary[w:]
    graph = {k: g[k] for k in ("A", "K", "edge_index") if k in g}     # keep the graph as stored (sparse stays sparse)
    return idx, {**graph, "raw_schedule": raw, "thresholded_schedule": binary}, gen_seconds


def parallel_rollout(model_path, graphs, params, workers):
    """Dual-roll every graph in its own worker (shared spawn-pool, one core each); returns (roll, times)."""
    desc = (f"SAGNN rollout (delta={params['delta']:g}, wd={params['weight_decay']:g}, "
            f"warmup={params['warmup_steps']}{', masked' if params['mask_collisions'] else ''})")
    return run_pool(graphs, init_fn=_init_rollout_worker, init_args=(model_path, params),
                    work_fn=_rollout_work, workers=workers, desc=desc)


def atomic_save(obj, path):
    """Save to a temp file then rename, so a failed/interrupted write (e.g. out of
    disk) never leaves a corrupt file at the destination."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description="Re-collect a UGNN dataset from a trained SAGNN policy")
    p.add_argument("--sagnn-model", default=None, help="trained SAGNN policy (models/exp_..._of_1.pt); not needed with --post-mask")
    p.add_argument("--graphs", required=True, help="expert_samples_*.pt to reuse conflict graphs + order from")
    p.add_argument("--out", required=True, help="output path for the new expert_samples_*.pt")
    p.add_argument("--delta", type=float, default=0.1, help="dual-update delta (training delta)")
    p.add_argument("--weight-decay", type=float, default=0.05, help="dual lambda regularization")
    p.add_argument("--T", type=int, default=200, help="dual-rollout steps per graph (all are rolled+timed)")
    p.add_argument("--warmup-steps", type=int, default=50,
                   help="leading cold-start steps DROPPED from the saved dataset, so the saved trajectory "
                        "is T - warmup_steps post-convergence slots (default 50). Timing still covers all T "
                        "rolled steps; to keep N saved slots use --T N+warmup_steps.")
    p.add_argument("--lr-lambda", type=float, default=2.0)
    p.add_argument("--T0", type=int, default=1)
    p.add_argument("--mask", action="store_true", help="mask the SAVED schedule (raw dual dynamics unchanged)")
    p.add_argument("--post-mask", action="store_true",
                   help="do NOT re-roll: mask the existing --graphs raw_schedule and keep its lambda "
                        "(lambda bit-identical to the unmasked dataset; SAGNN unchanged)")
    p.add_argument("--seed", type=int, default=0, help="base seed for collision tie-breaks (seed+graph_idx)")
    p.add_argument("--max-graphs", type=int, default=None, help="limit #graphs (quick test runs)")
    p.add_argument("--workers", type=int, default=min(24, (os.cpu_count() or 4) - 2),
                   help="parallel worker processes for the re-roll (one graph per worker, CPU). "
                        "1 = sequential on --device (cuda if available).")
    p.add_argument("--device", default=None, help="sequential (--workers 1) device; default: cuda if available else cpu")
    args = p.parse_args()

    graphs = load_graphs(args.graphs)
    if args.max_graphs:
        graphs = graphs[: args.max_graphs]

    times, workers = None, 0
    if args.post_mask:
        # Faithful masked dataset: mask the unmasked dataset's schedules in place,
        # keep lambda -> the SAGNN side is identical to the unmasked experiment.
        print(f"POST-MASK {len(graphs)} graphs from {args.graphs} (lambda kept unchanged)")
        roll = post_mask_dataset(graphs, seed=args.seed)        # data transform, not generation -> untimed
    else:
        if not args.sagnn_model:
            p.error("--sagnn-model is required unless --post-mask is given")
        workers = max(1, min(args.workers, len(graphs)))
        if workers > 1:
            print(f"parallel SAGNN rollout | graphs: {len(graphs)} | workers: {workers} (cpu) | "
                  f"delta={args.delta:g} wd={args.weight_decay:g} T={args.T} warmup={args.warmup_steps} mask={args.mask}")
            params = dict(T=args.T, lr_lambda=args.lr_lambda, delta=args.delta,
                          weight_decay=args.weight_decay, T0=args.T0, warmup_steps=args.warmup_steps,
                          mask_collisions=args.mask, seed=args.seed)
            roll, times = parallel_rollout(args.sagnn_model, graphs, params, workers)
        else:
            device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
            model = load_sagnn(args.sagnn_model, device=str(device))
            print(f"sequential SAGNN rollout | device: {device} | graphs: {len(graphs)} | mask={args.mask} | "
                  f"delta={args.delta:g} wd={args.weight_decay:g} T={args.T} warmup={args.warmup_steps}")
            roll, times = sagnn_dual_rollout(
                model, graphs, device, T=args.T, lr_lambda=args.lr_lambda, delta=args.delta,
                weight_decay=args.weight_decay, T0=args.T0, warmup_steps=args.warmup_steps,
                mask_collisions=args.mask, seed=args.seed, progress=True, return_time=True,
            )

    roll = [{k: v for k, v in g.items() if k != "lambda"} for g in roll]
    save_obj = {DEFAULT_SPLIT: roll}
    if times is not None:                                       # timing covers all T rolled steps per graph
        summary = summarize_times(times, args.T)
        print_timing_summary("SAGNN", summary, workers=workers, warmup_steps=args.warmup_steps)
        save_obj["meta"] = {"timing": timing_meta(
            "SAGNN", summary, workers=workers, schedules_per_graph=args.T,
            warmup_steps=args.warmup_steps, saved_per_graph=args.T - args.warmup_steps, delta=args.delta)}
    atomic_save(save_obj, args.out)
    print(f"saved {len(roll)} graphs -> {args.out}")


if __name__ == "__main__":
    main()
