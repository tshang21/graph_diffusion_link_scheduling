"""Samples files and the per-graph training batches.

A samples file is ``{DEFAULT_SPLIT: [entry, ...], "meta": {...}}``; each entry is a dict with
``A`` (K, K) conflict matrix, ``raw_schedule`` (T, K) schedules in [0, 1] one row per slot, and
optionally ``delta`` -- the min-rate target those schedules were produced for, whose presence turns
on conditioning (has_delta / delta_cond).  One graph rolled at several deltas is several entries.
A training example is a ``window``-slot block of one trajectory; iter_batches groups blocks into
batches of equal K.

    python -m ugnn.dataset slice --samples IN --out OUT --slot 28|random [--seed 0] [--slot-min 2]
    python -m ugnn.dataset tile  --samples IN --out OUT --tile 100

slice keeps ONE slot per graph (a fixed 1-based slot, or uniform on [slot-min, T]); tile repeats
each graph's slots N times so single-slot files train with full batches (no new information).
"""
import argparse
import collections
import os

import numpy as np
import torch
from torch_geometric.data import Batch, Data

DEFAULT_SPLIT = ("test", "per_graph")   # top-level key of the per-graph list (historical name)
TARGET_OFFSET = 0.5                     # x_0 = raw_schedule - 0.5, so the diffusion target lies in [-0.5, 0.5]


def load_samples(source, split=DEFAULT_SPLIT):
    """Per-graph list from a samples .pt path (or an already-loaded dict / list)."""
    obj = torch.load(source, map_location="cpu", weights_only=False) if isinstance(source, (str, os.PathLike)) else source
    return list(obj[split] if isinstance(obj, dict) else obj)

def save_samples(path, graphs, meta=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({DEFAULT_SPLIT: list(graphs), "meta": meta or {}}, path)


def graph_K(g):
    """Number of conflict-graph nodes."""
    return int(g["K"]) if "K" in g else int(np.asarray(g["A"]).shape[0])


def graph_edges(g):
    """(2, E) edge index, from the stored edges or from a dense A."""
    if "edge_index" in g:
        return torch.as_tensor(g["edge_index"], dtype=torch.long)
    return graph_A(g).nonzero().t().contiguous()


def graph_A(g):
    """Dense (K, K) adjacency, materialised from edge_index when the file stores edges only.

    A dense A is K^2 floats -- 100 MB at K=5000 -- so files for the larger graph families store
    ``edge_index`` + ``K`` instead. Scoring needs the dense form (one graph at a time); the
    training path must use graph_edges() and never call this."""
    if g.get("A") is not None:
        return torch.as_tensor(g["A"], dtype=torch.float32)
    K, ei = graph_K(g), graph_edges(g)
    A = torch.zeros(K, K)
    A[ei[0], ei[1]] = 1.0
    return A


def has_delta(graphs):
    """True when the samples carry a per-graph min-rate target -> train a delta-conditioned UGNN."""
    return "delta" in graphs[0]


DELTA_SCALE = 10.0                      # delta lives in ~[0, 0.2]; a constant input that small is
                                        # tiny next to normalised activations, so the encoder barely
                                        # separates 0.01 from 0.1.  Scaling to ~[0, 2] is a plain
                                        # change of input units, applied identically at train and
                                        # inference time (both go through delta_cond).


def delta_cond(delta, K, device=None):
    """(K,1,1) conditioning channel holding DELTA_SCALE*delta on every node, or None if delta is None.

    The denoiser's conditioning encoder takes one channel per node (data.x); delta is a
    per-graph scalar, so it is broadcast to a constant field over the conflict graph.
    """
    return None if delta is None else torch.full((K, 1, 1), DELTA_SCALE * float(delta), device=device)


def window_target(sched, start, window, K):
    """(K, 1, window) centred target: `window` CONSECUTIVE slots as the signal channels.

    window=1 is the i.i.d.-slot model.  window>1 makes one training example a short TRAJECTORY,
    so the denoiser sees how service rotates between slots -- the only place a min-rate target
    can show up, since delta constrains the long-run rate and not any single slot.  Put the
    window on the channel axis (not the UGNN time axis): the GNN layers mix channels, whereas
    the temporal axis needs gnn_config.temporal_mixer, which is off by default.
    """
    block = sched[start:start + window] - TARGET_OFFSET       # (window, K)
    return block.t().contiguous().view(K, 1, window)


def iter_batches(graphs, batch_size, shuffle, window=1, stride=None, windows=None, seed=0):
    """PyG batches of <= batch_size examples, grouped so every batch has ONE node count K.

    The denoiser is dense: ddpm.training_loss folds the flat node list back into [B, T, N, F] with
    ``N = data.y.size(0) // B``, so every item of a batch MUST have the same K -- mixing sizes would
    silently mis-shape every sample.  Within a K group, batching different graphs together is exact:
    the batched edge_index is block-diagonal so message passing never crosses graphs, and the model
    has no BatchNorm or other cross-sample statistic (verified by comparing a graph's output alone
    against the same graph inside a mixed batch).

    Blocks start every ``stride`` slots (default: stride = window, non-overlapping); stride=1
    slides, offering every possible start.  ``windows`` then keeps only that many blocks per
    trajectory, drawn ONCE from a per-graph seeded generator so the same subset is reused every
    epoch -- a genuinely smaller training set, not per-epoch augmentation.  They are drawn at
    random rather than taken from the front, which would train only on the cold start where the
    dual variables have not yet ramped.  Fewer, larger K-groups cost almost nothing extra: batch
    COUNT is set by the number of K groups, so raising ``windows`` fills batches the GPU was
    already launching (windows 1 -> 10 is 10x the data for 2.9x the time).
    An entry carrying a "delta" gets it as the conditioning channel on every item of its batches.
    """
    stride = window if stride is None else stride
    order = torch.randperm(len(graphs)).tolist() if shuffle else list(range(len(graphs)))
    by_K = collections.defaultdict(list)
    for gi in order:
        by_K[graph_K(graphs[gi])].append(gi)
    for K, gis in by_K.items():
        batch = []
        for gi in gis:
            g = graphs[gi]
            sched = torch.as_tensor(g["raw_schedule"], dtype=torch.float32)
            edge_index = graph_edges(g)          # never materialise a dense A on the training path
            cond = delta_cond(g.get("delta"), K)
            starts = list(range(0, len(sched) - window + 1, stride))
            if not starts:
                raise ValueError(f"graph has {len(sched)} slots < window {window}")
            if windows:
                gen = torch.Generator().manual_seed(seed * 1_000_003 + gi)
                starts = [starts[i] for i in sorted(torch.randperm(len(starts), generator=gen)[:windows].tolist())]
            if shuffle:
                starts = [starts[i] for i in torch.randperm(len(starts)).tolist()]
            for t in starts:
                batch.append(Data(edge_index=edge_index, x=cond, num_nodes=K,
                                  y=window_target(sched, t, window, K)))
                if len(batch) == batch_size:
                    yield Batch.from_data_list(batch)
                    batch = []
        if batch:
            yield Batch.from_data_list(batch)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sl, tl = sub.add_parser("slice"), sub.add_parser("tile")
    for s in (sl, tl):
        s.add_argument("--samples", required=True)
        s.add_argument("--out", required=True)
    sl.add_argument("--slot", required=True, help="1-based slot, or 'random'")
    sl.add_argument("--seed", type=int, default=0)
    sl.add_argument("--slot-min", type=int, default=2, help="smallest slot 'random' may draw (skips the near-max first solve)")
    tl.add_argument("--tile", type=int, required=True, help="copies of each graph's slots")
    args = p.parse_args()

    graphs, out, meta = load_samples(args.samples), [], vars(args)
    if args.cmd == "slice":
        rng = np.random.default_rng(args.seed)
        for g in graphs:
            T = g["raw_schedule"].shape[0]
            slot = int(rng.integers(args.slot_min, T + 1)) if args.slot == "random" else int(args.slot)
            out.append({k: v for k, v in g.items() if k == "delta"}
                       | {"A": g["A"], "raw_schedule": g["raw_schedule"][slot - 1: slot], "slot": slot})
        meta["slots"] = [g["slot"] for g in out]
    else:
        out = [{k: v for k, v in g.items() if k == "delta"}
               | {"A": g["A"], "raw_schedule": g["raw_schedule"].repeat(args.tile, 1)} for g in graphs]
    save_samples(args.out, out, meta)
    print(f"{args.cmd}: {len(out)} graphs x {out[0]['raw_schedule'].shape[0]} slots -> {args.out}")


if __name__ == "__main__":
    main()
