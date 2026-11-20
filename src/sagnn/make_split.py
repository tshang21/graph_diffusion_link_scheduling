"""Re-split the 250-graph SAWL dataset into a 150-train / 100-test experiment split.

The base file ``data/sawl_dataset_degree6_graphs_250.pt`` is a dict ``{train:50, test:200}``
of ``SAWLData`` (graphs only). This combines all 250, seeded-shuffles, and slices 150/100,
emitting three files:

  * ``data/sawl_dataset_degree6_split150.pt`` -- ``{train:[150 SAWLData], test:[100 SAWLData]}``
    for SAGNN training (``sagnn.collect_expert --dataset ...``; needs ``edge_index``).
  * ``new_results/graphs_train150.pt`` and ``new_results/graphs_test100.pt`` --
    ``{("test","per_graph"): [{"A": ...}, ...]}`` graph-only files for SAGNN rollout
    (``sagnn.collect_dataset --graphs ...``) and the comparison (``ugnn.compare_all --all-graphs``).

Run:
  PYTHONPATH=src python -m sagnn.make_split            # default seed 0, 150/100
"""
from __future__ import annotations

import argparse
import os
import random

import torch

import common.utils  # noqa: F401  registers the 'utils'/'SAWLData' alias for the pickle
from ugnn.dataset import DEFAULT_SPLIT


def main() -> None:
    p = argparse.ArgumentParser(description="Split the 250-graph SAWL dataset into 150 train / 100 test")
    p.add_argument("--source", default="data/sawl_dataset_degree6_graphs_250.pt",
                   help="base {train,test} SAWLData .pt (250 graphs)")
    p.add_argument("--n-train", type=int, default=150)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="shuffle seed (fixed -> reproducible disjoint split)")
    p.add_argument("--split-out", default="data/sawl_dataset_degree6_split150.pt",
                   help="output {train,test} SAWLData file for SAGNN training")
    p.add_argument("--graphs-dir", default="new_results",
                   help="dir for the graph-only graphs_train<N>.pt / graphs_test<N>.pt files")
    args = p.parse_args()

    data = torch.load(args.source, weights_only=False)
    allg = list(data["train"]) + list(data["test"])
    n = args.n_train + args.n_test
    if len(allg) < n:
        raise SystemExit(f"{args.source} has {len(allg)} graphs but split needs {n}")
    idx = list(range(len(allg)))
    random.Random(args.seed).shuffle(idx)
    train = [allg[i] for i in idx[: args.n_train]]
    test = [allg[i] for i in idx[args.n_train: args.n_train + args.n_test]]

    # 1) SAWLData split for SAGNN training (keeps edge_index/x/K/Delta).
    os.makedirs(os.path.dirname(os.path.abspath(args.split_out)) or ".", exist_ok=True)
    torch.save({"train": train, "test": test}, args.split_out)

    # 2) graph-only files (just A) for rollout + --all-graphs comparison.
    os.makedirs(args.graphs_dir, exist_ok=True)

    def _graph_only(graphs):
        return {DEFAULT_SPLIT: [{"A": torch.as_tensor(g["A"], dtype=torch.float32)} for g in graphs]}

    train_path = os.path.join(args.graphs_dir, f"graphs_train{args.n_train}.pt")
    test_path = os.path.join(args.graphs_dir, f"graphs_test{args.n_test}.pt")
    torch.save(_graph_only(train), train_path)
    torch.save(_graph_only(test), test_path)

    print(f"split seed={args.seed}: {len(train)} train / {len(test)} test (disjoint)")
    print(f"saved -> {args.split_out}")
    print(f"saved -> {train_path}")
    print(f"saved -> {test_path}")


if __name__ == "__main__":
    main()
