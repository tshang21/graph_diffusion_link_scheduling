"""One parallelism + one timing convention for every schedule generator.

The four generators -- SAGNN dual rollout, UGNN diffusion, greedy MIS, KaMIS
weighted MIS -- all produce ``T`` schedules per graph but on very different
substrates (GPU-capable torch, CPU numpy, an external C++ solver).  To compare
their *wall time* fairly they must share the same parallel strategy, so this
module is the single place that strategy lives:

  * **Same parallelism** -- a ``spawn`` process pool, one graph per task, one core
    per worker.  Each algorithm passes its own ``init_fn`` (loads per-worker state,
    e.g. a model or the solver path, and pins threads via ``torch.set_num_threads(1)``)
    and a module-level ``work_fn(task) -> (idx, result, gen_seconds)``.  Pass the same
    ``--workers`` to each algorithm and the comparison is apples-to-apples.
  * **Same timing** -- ``work_fn`` times ONLY its per-graph schedule loop (the caller
    decides exactly where that starts/stops) and returns the seconds; here we just
    aggregate.  Model load / pool spawn are outside the measured region by construction
    (init_fn runs once per worker, before any task).

GPU note: the two MIS methods are CPU-only (numpy greedy / external CPU solver), so a
fair head-to-head is CPU with equal ``--workers``.  Run the neural models on GPU only
when you want their best-case numbers, not for the cross-method comparison.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch.multiprocessing as mp
from tqdm import tqdm


def run_pool(graphs: Sequence, *, init_fn: Callable, init_args: tuple,
             work_fn: Callable, workers: int, desc: str, unit: str = "graph"):
    """Run ``work_fn`` over ``graphs`` in a spawn pool; return ``(results, times)`` in input order.

    ``init_fn(*init_args)`` runs once per worker (set up state, ``torch.set_num_threads(1)``).
    ``work_fn((idx, graph))`` must return ``(idx, result, gen_seconds)``.  ``times[i]`` is the
    per-graph schedule-loop seconds aligned with ``results[i]``.
    """
    tasks = list(enumerate(graphs))
    results = [None] * len(graphs)
    times = [None] * len(graphs)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=init_fn, initargs=init_args) as pool, \
            tqdm(total=len(graphs), desc=desc, unit=unit) as pbar:
        for idx, result, gen_seconds in pool.imap_unordered(work_fn, tasks, chunksize=1):
            results[idx] = result
            times[idx] = gen_seconds
            pbar.update(1)
    return results, times


def run_serial(graphs: Sequence, *, init_fn: Callable, init_args: tuple,
               work_fn: Callable, desc: str, unit: str = "graph"):
    """In-process equivalent of :func:`run_pool` (``--workers 1``): same init_fn/work_fn contract."""
    init_fn(*init_args)
    results = [None] * len(graphs)
    times = [None] * len(graphs)
    for task in tqdm(list(enumerate(graphs)), desc=desc, unit=unit):
        idx, result, gen_seconds = work_fn(task)
        results[idx] = result
        times[idx] = gen_seconds
    return results, times


def summarize_times(times: Sequence, schedules_per_graph, parallelism: int = 1) -> dict:
    """Aggregate per-graph gen seconds into throughput stats.

    ``times`` are per-graph single-unit compute seconds (one core / one sequential GPU
    graph).  ``parallelism`` is the method's native across-graph parallelism (CPU worker
    count, or 1 for a single GPU stream): the estimated wall-clock is ``sum(times)/parallelism``
    (the makespan of many similar jobs on ``parallelism`` units), so ``mean_seconds_per_graph``
    is the average wall time to produce one graph's schedules WITH that parallelism.
    ``schedules_per_graph`` is an int or a per-graph list.
    """
    times = [float(t) for t in times if t is not None]
    n = len(times)
    parallelism = max(1, int(parallelism))
    wall = float(sum(times)) / parallelism                     # est. wall-clock under native parallelism
    if isinstance(schedules_per_graph, (list, tuple)):
        n_sched = int(sum(schedules_per_graph))
    else:
        n_sched = n * int(schedules_per_graph)
    per_graph_sched = n_sched / n if n else 0.0
    mean_per_graph = wall / n if n else 0.0
    return {
        "num_graphs": n,
        "parallelism": parallelism,
        "schedules_total": n_sched,
        "total_gen_seconds": wall,
        "mean_seconds_per_graph": mean_per_graph,
        "mean_ms_per_schedule": 1000.0 * mean_per_graph / per_graph_sched if per_graph_sched else 0.0,
        "per_graph_seconds": times,
    }


def timing_meta(algo: str, summary: dict, *, workers: int, **extra) -> dict:
    """Comparable timing block to stash in an output's ``meta`` (per_graph_seconds dropped to keep it small)."""
    block = {k: v for k, v in summary.items() if k != "per_graph_seconds"}
    block.update(algo=algo, workers=workers, **extra)
    block["per_graph_seconds"] = summary.get("per_graph_seconds", [])
    return block


def print_timing_summary(algo: str, summary: dict, *, workers: int, **extra) -> None:
    """One-line wall-time summary for the console."""
    tail = "".join(f" | {k}={v}" for k, v in extra.items())
    print(f"  [{algo}] gen time | workers={workers} | "
          f"{summary['num_graphs']} graphs x {summary['schedules_total'] // max(summary['num_graphs'], 1)} sched"
          f"{tail} | total {summary['total_gen_seconds']:.2f}s | "
          f"{summary['mean_seconds_per_graph']:.4f} s/graph | "
          f"{summary['mean_ms_per_schedule']:.3f} ms/sched")
