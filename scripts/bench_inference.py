"""Inference cost of one schedule trajectory: MWIS teacher (1 CPU core) vs UGNN (1 GPU), per graph size and horizon.

    cd src && PYTHONPATH=. python ../scripts/bench_inference.py --method mwis   # CPU only, fine while the GPU trains
    cd src && PYTHONPATH=. python ../scripts/bench_inference.py --method ugnn   # needs an otherwise idle GPU

Both generate T slots at delta = 0.1 for the first --graphs test graphs of every size family and write
results/_bench/inference_<method>.json, one row per (size, T, graph).
MWIS  KaMIS.run_trajectory -- the collector's exact per-slot loop (write graph, sort, weighted local search
      capped at --time-limit, dual weight update) with the collector's settings (scale 1000, eta 2). One graph
      per spawn worker = one core each. ``seconds`` is the whole trajectory, ``solver_seconds`` only the
      weighted_local_search calls. MWIS is sequential in T: every slot's weights depend on the previous slot.
SAGNN the 500-link policy (scripts/run_sagnn_irregular500.sh; every size is zero-shot) in its dual rollout
      (sagnn.rollout.sagnn_dual_rollout_graph, wd 0.05, lr_lambda 2), one graph at a time on --device
      (cuda, or cpu = one thread) -> inference_sagnn_<device>.json. The rollout files' own meta timings were
      taken with 24 graphs sharing the CPU and are not comparable.
UGNN  the settled 500-link model (every size is zero-shot). A graph's windows go through the diffusion sampler
      in batches of up to --node-windows / K windows, so a longer horizon mostly fills the same calls.
      The first graph of every (size, T) runs twice and its first run is a discarded warm-up.
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import common.utils  # noqa: F401  SAWLData alias
from common.bench import run_pool
from MIS.KaMIS import DEFAULT_KAMIS_DIR, run_trajectory
from sagnn.rollout import load_sagnn, sagnn_dual_rollout_graph
from ugnn.dataset import graph_A, graph_edges, graph_K, load_samples
from ugnn.generate import generate_schedules
from ugnn.model import load_trained

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/delta_grid/irregular_lcc_noise0.02"
OUT = BASE / "results/_bench"
SIZES = (500, 1000, 2000, 5000)
DELTA = 0.1
SAGNN_MODEL = ROOT / ("src/models/exp_epochs_100_delta_0.1_lr_phi_5e-05_lr_lambda_2_weight_decay0.05_degree_6_of_1_"
                      "irregular_lcc_noise0.02.pt")
_W = {}


def test_graphs(L, n):
    src = BASE / "test_T100_delta0.1/samples_MWIS.pt" if L == 500 else ROOT / f"data/irregular_links{L}/graphs_test50.pt"
    return load_samples(src)[:n]


def _mwis_init(time_limit, workdir):
    torch.set_num_threads(1)
    _W.update(time_limit=time_limit, workdir=workdir)


def _mwis_work(task):
    idx, (g, T) = task
    with tempfile.TemporaryDirectory(dir=_W["workdir"]) as d:
        t0 = time.perf_counter()
        run_trajectory(g, 0, T, d, 1, DEFAULT_KAMIS_DIR, _W["time_limit"], 0, DELTA, 2.0,
                       clean=True, rule="dual", scale=1000.0)
        seconds = time.perf_counter() - t0
        solver = float(np.loadtxt(os.path.join(d, "mwis_times.tmp")).sum())
    return idx, solver, seconds


def bench_mwis(graphs, horizons, time_limit, workdir):
    rows = []
    for L, gs in graphs.items():
        for T in horizons:
            tasks = [SimpleNamespace(num_nodes=graph_K(g), edge_index=graph_edges(g)) for g in gs]
            solver, seconds = run_pool([(g, T) for g in tasks], init_fn=_mwis_init, init_args=(time_limit, workdir),
                                       work_fn=_mwis_work, workers=len(tasks), desc=f"MWIS {L} links T={T}")
            rows += [dict(links=L, T=T, graph=i, K=g.num_nodes, seconds=s, solver_seconds=v)
                     for i, (g, s, v) in enumerate(zip(tasks, seconds, solver))]
            print(f"MWIS {L:>5} links T={T:>5}: {np.mean(seconds):8.2f} s/graph "
                  f"(solver {np.mean(solver):8.2f} s)", flush=True)
    return rows


def bench_sagnn(graphs, horizons, device):
    device = torch.device(device)
    if device.type == "cpu":
        torch.set_num_threads(1)                              # one core, like the MWIS workers
    model = load_sagnn(str(SAGNN_MODEL), device=str(device))
    rows = []
    for L, gs in graphs.items():
        for T in horizons:
            for i, g in enumerate([gs[0], *gs]):              # gs[0] twice: the first run is the warm-up
                *_, sec = sagnn_dual_rollout_graph(model, graph_A(g), device, T=T, lr_lambda=2.0, delta=DELTA,
                                                   weight_decay=0.05, T0=1, return_time=True)
                if i:
                    rows.append(dict(links=L, T=T, graph=i - 1, K=graph_K(g), seconds=sec))
            print(f"SAGNN {L:>5} links T={T:>5}: {np.mean([r['seconds'] for r in rows[-len(gs):]]):8.2f} s/graph",
                  flush=True)
    return rows


def bench_ugnn(graphs, horizons, node_windows, sampling_timesteps, bf16):
    device = torch.device("cuda")
    diffusion = load_trained(str(next((BASE / "runs").glob("*ugnn_w10_bc512/*_ugnn_w10_bc512.pt"))), device,
                             sampling_timesteps=sampling_timesteps)
    rows = []
    for L, gs in graphs.items():
        for T in horizons:
            for i, g in enumerate([gs[0], *gs]):              # gs[0] twice: the first run is the warm-up
                torch.manual_seed(i)
                _, sec = generate_schedules(diffusion, graph_A(g), device, T,
                                            draw_batch=max(1, node_windows // graph_K(g)), delta=DELTA, bf16=bf16)
                if i:
                    rows.append(dict(links=L, T=T, graph=i - 1, K=graph_K(g), seconds=sec))
            print(f"UGNN {L:>5} links T={T:>5}: {np.mean([r['seconds'] for r in rows[-len(gs):]]):8.2f} s/graph",
                  flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["mwis", "ugnn", "sagnn"])
    p.add_argument("--device", default="cuda", help="SAGNN: cuda | cpu (one thread)")
    p.add_argument("--graphs", type=int, default=5, help="test graphs per size family")
    p.add_argument("--T", type=int, nargs="+", default=[10, 100, 1000], help="horizons (slots per trajectory)")
    p.add_argument("--time-limit", type=float, default=0.1, help="MWIS: solver seconds per slot (the collector's cap)")
    p.add_argument("--workdir", default=None, help="MWIS: directory for trajectory files (default: system temp)")
    p.add_argument("--node-windows", type=int, default=50_000,
                   help="UGNN: K x windows per sampler call (50k = the 10 windows the 5000-link rollouts used)")
    p.add_argument("--sampling-timesteps", type=int, default=50, help="UGNN: DDIM steps per sample call")
    p.add_argument("--bf16", action="store_true", help="UGNN: sample under bfloat16 autocast")
    args = p.parse_args()

    graphs = {L: test_graphs(L, args.graphs) for L in SIZES}
    if args.method == "mwis":
        rows = bench_mwis(graphs, args.T, args.time_limit, args.workdir)
        device = f"CPU, one core per graph ({os.cpu_count()} cores)"
    elif args.method == "sagnn":
        rows = bench_sagnn(graphs, args.T, args.device)
        device = torch.cuda.get_device_name() if args.device == "cuda" else "CPU, one thread, one graph at a time"
    else:
        rows = bench_ugnn(graphs, args.T, args.node_windows, args.sampling_timesteps, args.bf16)
        device = torch.cuda.get_device_name()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.device}" if args.method == "sagnn" else "" if args.method == "mwis" or (args.sampling_timesteps == 50 and not args.bf16) \
        else f"_s{args.sampling_timesteps}{'bf16' if args.bf16 else ''}"          # non-default sampler -> its own file
    path = OUT / f"inference_{args.method}{tag}.json"
    with open(path, "w") as f:
        json.dump(dict(meta=vars(args) | dict(delta=DELTA, device=device), rows=rows), f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
