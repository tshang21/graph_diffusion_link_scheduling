"""Generate deg6-recipe conflict graphs at arbitrary (larger) world sizes.

Same distribution as ``data/sawl_dataset_degree6_graphs_250.pt`` (see
``sagnn/data_tests.ipynb`` cell 11 / ``datageneration.create_conflict_dataset_irregular``):
``num_nodes = world_size**2`` users on a jittered grid (``noise * world_size``),
connected within radius ``r`` cells, line-graph dual -> conflict graph, kept only if the
average dual degree is within ``dual_tol`` of ``target_degree`` and the conflict graph is
connected.  Node density and radius-per-cell are size-invariant, so growing ``world_size``
scales K (~1.7 * num_nodes) while keeping the local geometry -- "same distribution, larger".
No target-K rejection: K falls where the size dictates.

Emits (into --out-dir) the same two file kinds as ``sagnn.make_split``:
  * ``sawl_split<n_train>.pt``      -- {train, test} SAWLData (edge_index/A/K; for SAGNN use)
  * ``graphs_train<n_train>.pt`` / ``graphs_test<n_test>.pt``
                                    -- {DEFAULT_SPLIT: [{"A": ...}]} graph-only files for
                                       src/MIS/run_pipeline.sh / ugnn.generate

Run (from src/, PYTHONPATH=.):
  python -m sagnn.make_graphs --world-sizes 23 24 25 --n-train 150 --n-test 50 \
      --out-dir ../data/deg6_w24
"""
from __future__ import annotations

import argparse
import math
import os

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph, RemoveIsolatedNodes

from common.utils import SAWLData, create_conflict_graph
from ugnn.dataset import DEFAULT_SPLIT


def sample_conflict_graph(world_size: int, r: float, noise: float, connect_mode: str = "lcc"):
    """One deg6-recipe conflict graph at the given world size (degree-unfiltered).

    connect_mode handles disconnected node graphs: the base recipe rejected them, but
    P(RGG fully connected) decays exponentially with area -- at w>=30 essentially every
    draw has some stray 2-3 node component, so rejection never terminates.  "lcc" (default)
    keeps the largest connected component of the NODE graph instead (its line graph -- the
    conflict graph -- is then connected by construction); the dropped-link fraction is
    returned so the caller can report it (~1-2% at these densities).  "reject" returns
    None for disconnected draws, reproducing the base behaviour.
    """
    num_nodes = world_size ** 2
    grid_dim = world_size
    cell = world_size / grid_dim  # = 1.0, kept for parity with the original construction

    centers = np.array([[(j + .5) * cell, (i + .5) * cell]
                        for i in range(grid_dim) for j in range(grid_dim)])
    idx = np.random.choice(len(centers), size=num_nodes, replace=False)
    positions = centers[idx] + np.random.normal(scale=noise * world_size, size=(num_nodes, 2))
    pos = torch.from_numpy(positions).float()
    x = torch.ones(num_nodes, 1, dtype=torch.long)

    radius = r * world_size / grid_dim
    remove_isolated = RemoveIsolatedNodes()
    try:
        data = remove_isolated(RadiusGraph(radius)(Data(x=x, edge_index=torch.tensor([], dtype=torch.long), pos=pos)))
    except Exception:
        # torch_cluster is broken in some envs -> scipy fallback with the same
        # dist <= radius semantics (symmetrized, no self-loops).
        from scipy.spatial import cKDTree
        pairs = cKDTree(positions).query_pairs(radius, output_type="ndarray")
        ei = (torch.from_numpy(np.concatenate([pairs, pairs[:, ::-1]]).T.copy()).long()
              if len(pairs) else torch.empty((2, 0), dtype=torch.long))
        data = remove_isolated(Data(x=x, edge_index=ei, pos=pos))

    G = nx.Graph()
    G.add_edges_from(data.edge_index.t().tolist())
    total_links = G.number_of_edges()
    if not nx.is_connected(G):
        if connect_mode == "reject":
            return None
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    dropped_frac = 1.0 - G.number_of_edges() / max(total_links, 1)
    conflict_data, A, K, len_edges, degrees = create_conflict_graph(graph=G)
    return conflict_data, A, K, degrees, dropped_frac


def main() -> None:
    p = argparse.ArgumentParser(description="Generate deg6-recipe conflict graphs at larger world sizes")
    p.add_argument("--world-sizes", type=int, nargs="+", required=True,
                   help="world sizes to sample uniformly per graph (num_nodes = w^2); "
                        "the base deg6 set used 15..19 -> K~500")
    p.add_argument("--n-train", type=int, default=150)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--r", type=float, default=1.2, help="connection radius in cells (deg6 recipe)")
    p.add_argument("--noise", type=float, default=0.01, help="grid jitter as a fraction of world size")
    p.add_argument("--target-degree", type=float, default=6.0, help="average dual-degree filter center")
    p.add_argument("--dual-tol", type=float, default=0.5, help="average dual-degree filter tolerance")
    p.add_argument("--deltavalue", type=float, default=0.1,
                   help="Delta stored on the SAWLData split file (downstream MWIS/UGNN pass their own delta)")
    p.add_argument("--connect-mode", default="lcc", choices=["lcc", "reject"],
                   help="disconnected node graphs: keep largest component (default; scales to any size) "
                        "or reject the draw (base recipe; never terminates for w>=~30)")
    p.add_argument("--sparse", action="store_true",
                   help="store edge_index + K instead of a dense A (required beyond K~2000, where "
                        "a dense A is hundreds of MB per graph); the MWIS pipeline reads edge_index "
                        "anyway, and ugnn.dataset.graph_A rebuilds a dense A on demand for scoring")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True, help="output folder (e.g. ../data/deg6_w24)")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_total = args.n_train + args.n_test
    graphs, Ks = [], []
    tried = 0
    while len(graphs) < n_total:
        tried += 1
        w = int(np.random.choice(args.world_sizes))
        sample = sample_conflict_graph(w, args.r, args.noise, connect_mode=args.connect_mode)
        if sample is None:
            continue
        conflict_data, A, K, degrees, dropped_frac = sample
        mean_deg = float(degrees.float().mean())
        if abs(mean_deg - args.target_degree) > args.dual_tol:
            continue
        Delta = torch.full((K, 1), args.deltavalue)
        x = torch.ones(K, dtype=torch.long).view(-1, 1)
        graphs.append(SAWLData(x=x, edge_index=conflict_data.edge_index,
                               A=None if args.sparse else A, Delta=Delta, K=K))
        Ks.append(K)
        print(f"[{len(graphs)}/{n_total}] w={w} K={K} mean_dual_degree={mean_deg:.2f} "
              f"dropped_links={dropped_frac:.1%} (accept rate {len(graphs)/tried:.0%})")

    train, test = graphs[: args.n_train], graphs[args.n_train:]
    os.makedirs(args.out_dir, exist_ok=True)

    split_path = os.path.join(args.out_dir, f"sawl_split{args.n_train}.pt")
    torch.save({"train": train, "test": test}, split_path)

    def _graph_only(gs):
        # dense A is K^2 floats (100 MB/graph at K=5000); --sparse stores the edges instead
        if args.sparse:
            return {DEFAULT_SPLIT: [{"edge_index": g["edge_index"].clone(), "K": int(g["K"])}
                                    for g in gs]}
        return {DEFAULT_SPLIT: [{"A": torch.as_tensor(g["A"], dtype=torch.float32)} for g in gs]}

    train_path = os.path.join(args.out_dir, f"graphs_train{args.n_train}.pt")
    test_path = os.path.join(args.out_dir, f"graphs_test{args.n_test}.pt")
    torch.save(_graph_only(train), train_path)
    torch.save(_graph_only(test), test_path)

    Ks = np.asarray(Ks)
    print(f"done: {n_total} graphs (train {args.n_train} / test {args.n_test}), "
          f"K min/mean/max = {Ks.min()}/{Ks.mean():.0f}/{Ks.max()}")
    print(f"saved -> {split_path}")
    print(f"saved -> {train_path}")
    print(f"saved -> {test_path}")


if __name__ == "__main__":
    main()
