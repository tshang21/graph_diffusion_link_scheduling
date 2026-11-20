"""File-based MWIS trajectory pipeline (Romi's implementation, adapted to this repo).

Driven by ``run_pipeline.sh``: per graph, ``init`` writes the METIS (fmt=10) graph
file with uniform starting node weights; KaMIS' ``sort_adjacencies`` +
``weighted_local_search`` solve one MWIS per timestep; ``evolve`` reads the solver's
0/1 membership file and updates the node weights for the next slot.  Two rules, both
holding a REAL-valued per-link urgency and discretizing only at write time (--weight-rule):

  dual     (default) DUAL DYNAMICS of "Long-Horizon Wireless Link Scheduling with State-
           Augmented Graph Neural Networks" (arXiv:2607.18480), eq. (21), with lam(1) = 0:

               lam <- [ lam - eta * ( s . [1 - As]_+  -  Delta ) ]_+

           per link:  served -> lam - eta*(1 - delta);   starved -> lam + eta*delta.
           ``s . [1 - As]_+`` is the SUCCESSFUL-transmission indicator, which equals s on an
           independent set, so for MWIS slots it is exactly the served flag.  lam is the
           multiplier on rate_i >= delta, carried real-valued and projected onto lam >= 0
           each step, so a link that runs ahead forgets its surplus at once.  The node
           weight is (1 + lam) -- the coefficient of s in the paper's Lagrangian eq. (4),
           M(s, lam) = (1 + lam)' [s . [1-As]_+] - lam' Delta.  The paper uses eta = 2.

  deficit  the earlier NON-dual rule, kept for reference (evolve_weights_deficit):
               w_i = base + eta * max(0, delta * (slot + 1) - times_scheduled_i)
           recomputed from the service counts each slot, so a link far ahead of quota must
           "pay back" its whole surplus before it becomes urgent again.

Both keep their state REAL and hand ``discretize`` a real weight vector, because KaMIS node
weights MUST be positive integers: ``sort_adjacencies`` parses "5.5 2" as weight 5 and then
DROPS that node's adjacency list, so a decimal weight silently yields an edgeless graph (the
solver then dies with "number of specified edges mismatch").  ``discretize`` multiplies every
weight by ``--weight-scale`` before rounding; scaling ALL weights by one constant leaves the
MWIS argmax unchanged, so this is exact up to a resolution of 1/scale -- which is what lets a
dual step of eta*delta = 0.2 survive as an integer weight.

Modes (state lives in ``--out_dir`` files):
  run     T-slot trajectories for a graph RANGE in one invocation: the dataset is
          loaded once, graphs fan out across a spawn pool (--workers), and each
          graph runs init -> (sort_adjacencies -> weighted_local_search -> evolve) x T
          in-process.  Same files, binaries and weight rule as the step modes below,
          minus the ~3.6s python+torch startup per timestep and the dataset reload
          per graph.  --graph_index runs a single graph; else --graph_start/--graph_end.
  init    graph_{g}_t0.gr + weights_{g}_t0.pt + edges_{g}.pt (structure cache so
          evolve never reloads the full dataset)
  evolve  mwis_{g}_t{t}.out + weights_{g}_t{t}.pt -> graph/weights for t+1
  export  stack mwis_{g}_t*.out into samples_MWIS.pt (the ``ugnn.dataset``
          samples-file format -> drops into ugnn.train)
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

# repo datasets pickle classes from common.utils under the old 'utils' name;
# make both importable no matter the CWD this script is launched from.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
import common.utils  # noqa: F401  registers the 'utils' alias for dataset pickles
from common.bench import run_pool, run_serial
from ugnn.dataset import save_samples

DEFAULT_DATASET = os.path.join("data", "sawl_dataset_degree6_split150.pt")
DEFAULT_OUT_DIR = os.path.join("data", "mwis_pipeline")
DEFAULT_KAMIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "external", "KaMIS", "deploy")


def load_dataset(dataset_path, split="test"):
    try:
        # mmap keeps tensor storages file-backed: the multi-GB dataset files load in
        # <1 GB RSS (only pages actually touched -- edge_index, per-graph A -- come in)
        data = torch.load(dataset_path, weights_only=False, mmap=True)
    except Exception:
        data = torch.load(dataset_path, weights_only=False)
    return data[split]


def write_dimacs_graph(graph, node_weights, out_path):
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index

    adj_list = [[] for _ in range(num_nodes)]
    edge_set = set()

    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u == v:
            continue
        if (v, u) not in edge_set:
            adj_list[u].append(v)
            adj_list[v].append(u)
            edge_set.add((u, v))

    with open(out_path, 'w') as f:
        f.write(f"{num_nodes} {len(edge_set)} 10\n")
        for i in range(num_nodes):
            neighbors = [str(n + 1) for n in sorted(adj_list[i])]
            f.write(f"{node_weights[i]} {' '.join(neighbors)}\n")


def discretize(weights, scale):
    """Real node weights -> the positive integers KaMIS requires (the ONLY rounding point).

    Multiplying every weight by the same constant does not change which set maximises the
    total weight, so scaling then rounding is faithful up to a resolution of 1/scale.
    """
    return (torch.as_tensor(weights, dtype=torch.float32) * scale).round().clamp(min=1).int()


def evolve_weights_dual_dynamics(lam, scheduled_nodes, delta, eta, base=1.0):
    """Dual dynamics, arXiv:2607.18480 eq. (21); returns (lam, REAL weights = base + lam).

        served   lam <- [ lam - eta * (1 - delta) ]_+
        starved  lam <- [ lam + eta * delta ]_+

    ``eta`` is the dual step size (the paper uses 2).  MWIS slots are independent sets, so
    the paper's successful-transmission term s . [1 - As]_+ is exactly the served flag.
    """
    step = torch.full_like(lam, eta * delta)          # starved links climb
    step[scheduled_nodes] = -eta * (1.0 - delta)      # served links fall
    lam = (lam + step).clamp(min=0.0)
    return lam, base + lam


def evolve_weights_deficit(schedule_memory, scheduled_nodes, slot, delta, eta, base=1.0):
    """Earlier NON-dual rule, kept for reference; returns (schedule_memory, REAL weights).

    urgency = max(0, delta*(slot+1) - times_scheduled), recomputed from the running service
    counts (updated in place with this slot's winners) rather than accumulated.
    """
    schedule_memory[scheduled_nodes] += 1
    urgency = (delta * (slot + 1) - schedule_memory).clamp(min=0)
    return schedule_memory, base + eta * urgency


def graph_delta(graph_idx, delta, delta_max, seed):
    """Fixed ``delta``, or a per-graph draw from U(0, delta_max) when ``delta_max`` is set.

    Keyed on the graph index (not the pool order), so a graph gets the same delta whatever the
    worker count.  ``run_trajectory`` writes it next to the .out files for ``export`` to pick up.
    """
    if not delta_max:
        return float(delta)
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(graph_idx))
    return float(torch.rand(1, generator=g).item() * delta_max)


def _load_graph_struct(graph_idx, out_dir):
    """The (num_nodes, edge_index) cache written by init -- evolve runs once per
    timestep in a fresh process, so it must not pay the full dataset load."""
    struct = torch.load(os.path.join(out_dir, f"edges_{graph_idx}.pt"), weights_only=True)
    return SimpleNamespace(num_nodes=int(struct["num_nodes"]), edge_index=struct["edge_index"])


def init(graph_idx, out_dir, dataset_path, split, init_weight):
    test_data = load_dataset(dataset_path, split)
    graph = test_data[graph_idx]
    # weights must be non-negative integers for KaMIS (fmt=10 node weights)
    weights = torch.full((graph.num_nodes,), init_weight, dtype=torch.int)
    out_path = os.path.join(out_dir, f"graph_{graph_idx}_t0.gr")
    write_dimacs_graph(graph, weights, out_path)
    torch.save(weights, os.path.join(out_dir, f"weights_{graph_idx}_t0.pt"))
    torch.save({"num_nodes": graph.num_nodes, "edge_index": graph.edge_index},
               os.path.join(out_dir, f"edges_{graph_idx}.pt"))


def evolve(graph_idx, t, out_dir, delta, eta):
    graph = _load_graph_struct(graph_idx, out_dir)

    prev_weights_path = os.path.join(out_dir, f"weights_{graph_idx}_t{t}.pt")
    weights = torch.load(prev_weights_path, weights_only=True)

    mwis_path = os.path.join(out_dir, f"mwis_{graph_idx}_t{t}.out")
    with open(mwis_path, 'r') as f:
        included_flags = list(map(int, f.read().strip().split()))
    if len(included_flags) != graph.num_nodes:
        raise RuntimeError(f"{mwis_path}: {len(included_flags)} membership lines, "
                           f"expected {graph.num_nodes}")
    mwis_nodes = [i for i, included in enumerate(included_flags) if included == 1]

    # step mode runs one slot per process, so the counter is rebuilt from slots 0..t-1
    # (the mwis_*.out files ARE the state); evolve_weights then adds slot t itself.
    schedule_memory = torch.zeros(graph.num_nodes, dtype=torch.int)
    for prev in range(t):
        with open(os.path.join(out_dir, f"mwis_{graph_idx}_t{prev}.out")) as f:
            schedule_memory += torch.tensor([int(line) for line in f if line.strip()], dtype=torch.int)

    _, real_w = evolve_weights_deficit(schedule_memory, mwis_nodes, t + 1, delta, eta)
    new_weights = discretize(real_w, 1.0)

    out_path = os.path.join(out_dir, f"graph_{graph_idx}_t{t+1}.gr")
    write_dimacs_graph(graph, new_weights, out_path)
    torch.save(new_weights, os.path.join(out_dir, f"weights_{graph_idx}_t{t+1}.pt"))


def _check_binaries(kamis_dir):
    sort_bin = os.path.join(kamis_dir, "sort_adjacencies")
    wls_bin = os.path.join(kamis_dir, "weighted_local_search")
    for b in (sort_bin, wls_bin):
        if not (os.path.isfile(b) and os.access(b, os.X_OK)):
            raise FileNotFoundError(f"KaMIS binary missing: {b}")
    return sort_bin, wls_bin


def run_trajectory(graph, graph_idx, T, out_dir, init_weight, kamis_dir, time_limit, seed, delta, eta,
                   clean=False, rule="dual", scale=100.0):
    """The full T-slot trajectory for one (already loaded) graph in-process.

    Same on-disk protocol as the step-wise init/evolve loop (every graph_*.gr /
    *_sorted.gr / mwis_*.out / weights_*.pt still written, solver seconds appended
    to mwis_times.tmp) -- only the per-timestep python startup is gone.  ``graph``
    just needs ``.num_nodes`` and ``.edge_index``.
    """
    sort_bin, wls_bin = _check_binaries(kamis_dir)
    with open(os.path.join(out_dir, f"delta_{graph_idx}.txt"), "w") as f:
        f.write(f"{delta!r}\n")                    # per-graph delta -> export stores it in the samples
    # slot 0 has no history under either rule: urgency 0 everywhere -> uniform base weights,
    # i.e. a plain maximum independent set.  ``init_weight`` is that base.
    state = torch.zeros(graph.num_nodes, dtype=torch.float32 if rule == "dual" else torch.int)
    weights = discretize(torch.full((graph.num_nodes,), float(init_weight)), scale)
    write_dimacs_graph(graph, weights, os.path.join(out_dir, f"graph_{graph_idx}_t0.gr"))
    torch.save(weights, os.path.join(out_dir, f"weights_{graph_idx}_t0.pt"))

    times_path = os.path.join(out_dir, "mwis_times.tmp")
    for t in range(T):
        infile = os.path.join(out_dir, f"graph_{graph_idx}_t{t}.gr")
        mwis_path = os.path.join(out_dir, f"mwis_{graph_idx}_t{t}.out")

        # weighted_local_search INFINITE-LOOPS on weight-0 vertices (confirmed: the
        # same input solves instantly once the zeros are gone). Zero-weight nodes can
        # never increase an MWIS objective, so solve the positive-weight subgraph and
        # mark the zeros unscheduled -- identical optimum, no hang.
        zeros = weights == 0
        has_zeros = bool(zeros.any())
        if has_zeros:
            keep = (~zeros).nonzero().flatten()
            remap = torch.full((graph.num_nodes,), -1, dtype=torch.long)
            remap[keep] = torch.arange(keep.numel())
            ei = graph.edge_index
            em = (~zeros[ei[0]]) & (~zeros[ei[1]])
            sub = SimpleNamespace(num_nodes=int(keep.numel()), edge_index=remap[ei[:, em]])
            solve_in = os.path.join(out_dir, f"graph_{graph_idx}_t{t}_pos.gr")
            write_dimacs_graph(sub, weights[keep], solve_in)
            solve_out = solve_in + ".out"
        else:
            solve_in, solve_out = infile, mwis_path
        sorted_path = solve_in[:-3] + "_sorted.gr"

        with open(sorted_path, "w") as f:
            subprocess.run([sort_bin, solve_in], stdout=f, check=True)
        start = time.perf_counter()
        
        for attempt in range(5):
            try:
                subprocess.run([wls_bin, sorted_path, f"--output={solve_out}",
                                f"--time_limit={time_limit}", f"--seed={seed + 1000 * attempt}"],
                               stdout=subprocess.DEVNULL, check=True, timeout=60)
                break
            except subprocess.TimeoutExpired:
                print(f"[graph {graph_idx} t{t}] solver hung >60s, retrying with seed "
                      f"{seed + 1000 * (attempt + 1)}", flush=True)
        else:
            raise RuntimeError(f"graph {graph_idx} t{t}: weighted_local_search hung on 5 seeds")
        with open(times_path, "a") as f:
            f.write(f"{time.perf_counter() - start:.9f}\n")

        with open(solve_out) as f:
            included_flags = list(map(int, f.read().strip().split()))
        if has_zeros:
            if len(included_flags) != keep.numel():
                raise RuntimeError(f"{solve_out}: {len(included_flags)} membership lines, "
                                   f"expected {keep.numel()} positive-weight nodes")
            full = [0] * graph.num_nodes            # zero-weight nodes -> unscheduled
            for pos, node in enumerate(keep.tolist()):
                full[node] = included_flags[pos]
            included_flags = full
            with open(mwis_path, "w") as f:         # full-length flags for export/evolve
                f.write("\n".join(map(str, included_flags)) + "\n")
        if len(included_flags) != graph.num_nodes:
            raise RuntimeError(f"{mwis_path}: {len(included_flags)} membership lines, "
                               f"expected {graph.num_nodes}")
        mwis_nodes = [i for i, inc in enumerate(included_flags) if inc == 1]
        if rule == "dual":
            state, real_w = evolve_weights_dual_dynamics(state, mwis_nodes, delta, eta, init_weight)
        else:
            state, real_w = evolve_weights_deficit(state, mwis_nodes, t + 1, delta, eta, init_weight)
        weights = discretize(real_w, scale)

        write_dimacs_graph(graph, weights, os.path.join(out_dir, f"graph_{graph_idx}_t{t+1}.gr"))
        torch.save(weights, os.path.join(out_dir, f"weights_{graph_idx}_t{t+1}.pt"))

    if clean:
        # keep only the mwis_*.out membership files (all export needs); at K~4000+
        # the accumulated .gr/weights intermediates would otherwise reach many GB
        for pat in (f"graph_{graph_idx}_t*.gr", f"graph_{graph_idx}_t*_pos.gr.out",
                    f"weights_{graph_idx}_t*.pt"):
            for f in glob.glob(os.path.join(out_dir, pat)):
                os.remove(f)


# ── parallel fan-out (spawn pool; per-graph files are disjoint, so no locking) ──
_W = {}  # per-worker params


def _init_run_worker(params):
    torch.set_num_threads(1)   # one core per worker (the solver subprocess is single-threaded)
    _W["p"] = params


def _run_work(task):
    pos, (graph_idx, num_nodes, edge_index, delta) = task
    p = _W["p"]
    g = SimpleNamespace(num_nodes=num_nodes, edge_index=edge_index)
    t0 = time.perf_counter()
    run_trajectory(g, graph_idx, p["T"], p["out_dir"], p["init_weight"],
                   p["kamis_dir"], p["time_limit"], p["seed"], delta, p["eta"], clean=p["clean"],
                   rule=p["rule"], scale=p["scale"])
    return pos, graph_idx, time.perf_counter() - t0


def run_graphs(graph_start, graph_end, T, out_dir, dataset_path, split, init_weight,
               kamis_dir, time_limit, seed, delta, eta, workers, clean=False, delta_max=0.0,
               rule="dual", scale=100.0):
    """Trajectories for graphs [graph_start..graph_end] (dataset indices, inclusive).

    The dataset is loaded ONCE here; workers receive only each graph's
    (num_nodes, edge_index).  ``workers>1`` fans graphs across a spawn pool
    (the shared ``common.bench`` harness -- same as the other collectors).
    """

    _check_binaries(kamis_dir)                 # fail fast before the load + pool spin-up
    data = load_dataset(dataset_path, split)
    if not (0 <= graph_start <= graph_end < len(data)):
        raise ValueError(f"graph range [{graph_start}, {graph_end}] outside split of {len(data)} graphs")
    tasks = [(gi, int(data[gi].num_nodes), data[gi].edge_index, graph_delta(gi, delta, delta_max, seed))
             for gi in range(graph_start, graph_end + 1)]
    if delta_max:
        ds = [t[3] for t in tasks]
        print(f"per-graph delta ~ U(0, {delta_max:g}): min {min(ds):.4f}, mean {sum(ds)/len(ds):.4f}, max {max(ds):.4f}")
    params = dict(T=T, out_dir=out_dir, init_weight=init_weight, kamis_dir=kamis_dir,
                  time_limit=time_limit, seed=seed, eta=eta, clean=clean,
                  rule=rule, scale=scale)
    workers = max(1, min(workers, len(tasks)))
    desc = f"MWIS trajectories T={T} (x{workers})"
    kw = dict(init_fn=_init_run_worker, init_args=(params,), work_fn=_run_work, desc=desc)
    if workers > 1:
        run_pool(tasks, workers=workers, **kw)
    else:
        run_serial(tasks, **kw)


def export(out_dir, dataset_path, split, T, delta, eta, init_weight, store_A=True, rule="dual", scale=100.0):
    """Stack every graph's mwis_{g}_t*.out into a UGNN-trainable samples_MWIS.pt.

    MWIS slots are independent sets by construction (collision-free), so the raw
    stacked trajectory IS the masked training target -- no collision repair needed.
    """
    graph_ids = sorted(int(m.group(1)) for f in glob.glob(os.path.join(out_dir, "mwis_*_t0.out"))
                       if (m := re.search(r"mwis_(\d+)_t0\.out$", f)))
    if not graph_ids:
        raise FileNotFoundError(f"no mwis_*_t0.out files in {out_dir}; run the pipeline first")

    data = load_dataset(dataset_path, split)
    trajs, graphs = [], []
    for g in graph_ids:
        rows = []
        for t in range(T):
            path = os.path.join(out_dir, f"mwis_{g}_t{t}.out")
            with open(path) as f:
                rows.append([int(line) for line in f if line.strip()])
        traj = torch.tensor(rows, dtype=torch.float32)          # (T, K)
        K = data[g].num_nodes
        if traj.shape[1] != K:
            raise RuntimeError(f"graph {g}: solver rows have {traj.shape[1]} nodes, dataset says {K}")
        trajs.append(traj)
        graphs.append(data[g])

    # the delta each trajectory was actually rolled at (run_trajectory wrote it per graph);
    # with --delta-max these differ per graph and become the UGNN's conditioning signal.
    deltas = []
    for g in graph_ids:
        f = os.path.join(out_dir, f"delta_{g}.txt")
        deltas.append(float(open(f).read()) if os.path.exists(f) else float(delta))

    meta = {"source": "MWIS", "delta": delta, "eta": eta, "init_weight": init_weight, "T": T,
            "weight_rule": rule, "weight_scale": scale,
            "n_graphs": len(graphs), "store_A": store_A,
            "delta_min": min(deltas), "delta_max": max(deltas), "per_graph_delta": len(set(deltas)) > 1,
            "source_impl": f"MIS.KaMIS file pipeline ({rule} weights, base {init_weight} + round(eta*urgency))",
            "dataset": dataset_path, "split": split, "graph_ids": graph_ids}
    # store_A=False: at K~4000+ the dense per-graph A matrices would make the file tens of GB;
    # consumers then take A from the dataset file itself.
    per_graph = [({"A": g["A"].float(), "raw_schedule": traj} if store_A
                  else {"raw_schedule": traj, "K": int(traj.shape[1])}) | {"delta": d}
                 for g, traj, d in zip(graphs, trajs, deltas)]
    path = os.path.join(out_dir, "samples_MWIS.pt")
    save_samples(path, per_graph, meta)
    print(f"exported {len(graphs)} graphs x {T} slots -> {path} (store_A={store_A}, "
          f"delta {min(deltas):.4f}-{max(deltas):.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["run", "init", "evolve", "export"], required=True)
    parser.add_argument("--graph_index", type=int, help="single graph id within the split (run/init/evolve)")
    parser.add_argument("--graph_start", type=int, default=None, help="first graph id, inclusive (run mode)")
    parser.add_argument("--graph_end", type=int, default=None, help="last graph id, inclusive (run mode)")
    parser.add_argument("--workers", type=int, default=min(24, (os.cpu_count() or 4) - 2),
                        help="parallel worker processes, one graph/core each (run mode)")
    parser.add_argument("--t", type=int, help="current timestep (only for evolve mode)")
    parser.add_argument("--T", type=int, default=100, help="slots per graph (run/export modes)")
    parser.add_argument("--kamis_dir", type=str, default=DEFAULT_KAMIS_DIR,
                        help="dir holding sort_adjacencies + weighted_local_search (run mode)")
    parser.add_argument("--time_limit", type=float, default=0.01,
                        help="KaMIS anytime-search budget per slot in seconds (run mode)")
    parser.add_argument("--seed", type=int, default=0, help="KaMIS RNG seed (run mode)")
    parser.add_argument("--clean", action="store_true",
                        help="run mode: delete each graph's .gr/weights intermediates after its "
                             "trajectory (keeps mwis_*.out; big-K runs need this to bound disk)")
    parser.add_argument("--no-store-A", dest="store_A", action="store_false", default=True,
                        help="export mode: omit the dense per-graph A matrices (schedules only)")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--init_weight", type=int, default=1,
                        help="uniform starting node weight (init mode)")
    parser.add_argument("--weight-rule", dest="weight_rule", default="dual", choices=["dual", "deficit"],
                        help="dual: arXiv:2607.18480 eq.(21) dual dynamics, weight (1 + lam); "
                             "deficit: max(0, delta*(t+1) - times_scheduled) (the earlier rule)")
    parser.add_argument("--weight-scale", dest="weight_scale", type=float, default=100.0,
                        help="integer discretization: w_int = round(scale * w_real). Scaling every weight "
                             "alike leaves the MWIS argmax unchanged, so this only sets the resolution "
                             "(1/scale); KaMIS silently corrupts decimal weights.")
    parser.add_argument("--delta-max", dest="delta_max", type=float, default=0.0,
                        help="run mode: draw each graph's delta from U(0, this) instead of using --delta "
                             "(0 = off). The draw is keyed on the graph index and stored per graph in the "
                             "exported samples, so a delta-conditioned UGNN can train on the mix.")
    parser.add_argument("--delta", type=float, default=0.1,
                        help="minimum constraint")
    parser.add_argument("--eta", type=float, default=2.0,
                        help="priority scale on the paced deficit, w = 1 + eta*max(0, delta*(t+1) - served) "
                             "(run/evolve; recorded in the samples' meta). Stable for eta >= 2.")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "run":
        if args.graph_index is not None:
            start = end = args.graph_index
        elif args.graph_start is not None and args.graph_end is not None:
            start, end = args.graph_start, args.graph_end
        else:
            raise ValueError("run mode needs --graph_index or --graph_start + --graph_end")
        run_graphs(start, end, args.T, args.out_dir, args.dataset, args.split,
                   args.init_weight, args.kamis_dir, args.time_limit, args.seed, args.delta,
                   args.eta, args.workers, clean=args.clean, delta_max=args.delta_max,
                   rule=args.weight_rule, scale=args.weight_scale)
    elif args.mode == "init":
        if args.graph_index is None:
            raise ValueError("--graph_index must be specified for init mode")
        init(args.graph_index, args.out_dir, args.dataset, args.split, args.init_weight)
    elif args.mode == "evolve":
        if args.graph_index is None or args.t is None:
            raise ValueError("--graph_index and --t must be specified for evolve mode")
        evolve(args.graph_index, args.t, args.out_dir, args.delta, args.eta)
    elif args.mode == "export":
        export(args.out_dir, args.dataset, args.split, args.T, args.delta, args.eta, args.init_weight,
               store_A=args.store_A, rule=args.weight_rule, scale=args.weight_scale)
