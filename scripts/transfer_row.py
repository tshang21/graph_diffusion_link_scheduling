"""Roll one trained model on every size family and swap it into the size-transfer matrix.

    cd src && PYTHONPATH=. python ../scripts/transfer_row.py --model 2000 --ckpt <path> --archive-tag e500

Writes results/_transfer_cache/m<model>e<epoch>_on<L>_d<delta>.npz for every test family L and delta,
reusing cells the extension checks already rolled for the same checkpoint. It then moves the current canonical m<model>_* cells to m<model><archive-tag>_* and
promotes the new ones. Safe to re-run: finished cells are skipped, the swap refuses to run with cells
missing, and an existing archive is never overwritten.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

import common.utils  # noqa: F401  SAWLData alias
from ugnn.dataset import graph_K, load_samples
from ugnn.generate import rollout_ugnn
from ugnn.metrics import aggregate_method

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/delta_grid/irregular_lcc_noise0.02"
TC, EXT = BASE / "results/_transfer_cache", BASE / "results/_extension_cache"
SIZES, DELTAS = ("500", "1000", "2000", "5000"), (0.1, 0.05, 0.01)


def testset(L):
    if L == "500":
        return load_samples(BASE / "test_T100_delta0.1/samples_MWIS.pt")
    return load_samples(ROOT / f"data/irregular_links{L}/graphs_test50.pt")


def teacher_obj(L, d):
    f = (BASE / f"test_T100_delta{d:g}" if L == "500" else ROOT / f"data/irregular_links{L}/test_T100_delta{d:g}") / "samples_MWIS.pt"
    return float(np.mean([100 * np.asarray(x["raw_schedule"], np.float32).mean(0).mean() for x in load_samples(f)]))


def summary(m, prefix):
    for d in DELTAS:
        z = [np.load(TC / f"{prefix}_on{L}_d{d:g}.npz") for L in SIZES]
        pct = np.mean([100 * float(x["obj"].mean()) / teacher_obj(L, d) for x, L in zip(z, SIZES)])
        viol = np.mean([100 * float((x["rates"] < d - 1e-9).mean()) for x in z])
        print(f"  δ={d:<4g} {prefix:>12}: {pct:.2f}% of MWIS | {viol:.2f}% links below δ", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="training size of the model: 500|1000|2000|5000")
    p.add_argument("--ckpt", required=True, help="checkpoint path of the model to roll")
    p.add_argument("--archive-tag", required=True, help="suffix for the replaced cells, e.g. e500")
    p.add_argument("--dry-run", action="store_true", help="print the plan, roll and move nothing")
    args = p.parse_args()
    m = args.model
    ep = int(torch.load(args.ckpt, map_location="cpu", weights_only=False)["epoch"]) + 1
    new = lambda L, d: TC / f"m{m}e{ep}_on{L}_d{d:g}.npz"
    cur = lambda L, d: TC / f"m{m}_on{L}_d{d:g}.npz"
    arch = lambda L, d: TC / f"m{m}{args.archive_tag}_on{L}_d{d:g}.npz"
    cells = [(L, d) for d in DELTAS for L in SIZES]

    if all(arch(L, d).exists() for L, d in cells) and not any(new(L, d).exists() for L, d in cells):
        print(f"model {m}: already swapped (archive m{m}{args.archive_tag}_* complete)")
        return
    todo = [(L, d) for L, d in cells if not new(L, d).exists()]
    reusable = [(L, d) for L, d in todo if (EXT / f"m{m}_ep{ep}_on{L}_d{d:g}.npz").exists()]
    print(f"model {m} @ epoch {ep}: {len(todo)} cells to produce ({len(reusable)} reused from the extension "
          f"evaluation); current m{m}_* cells will be archived as m{m}{args.archive_tag}_*", flush=True)
    if args.dry_run:
        for L, d in todo:
            print(f"    {'reuse' if (L, d) in reusable else 'roll '}  test {L:>4} δ={d:g}")
        return

    graphs = {}
    for L, d in todo:
        gs = graphs.setdefault(L, testset(L))
        Ks = np.array([graph_K(g) for g in gs])
        if (L, d) in reusable:
            z = np.load(EXT / f"m{m}_ep{ep}_on{L}_d{d:g}.npz")
            np.savez(new(L, d), rates=z["rates"], obj=z["obj"], Ks=Ks)
            print(f"  reused extension cell -> test {L} δ={d:g}", flush=True)
            continue
        sch, sec = rollout_ugnn(args.ckpt, gs, T=100, device="cuda", seed=0, delta=d)
        a = aggregate_method(sch, gs, d)
        r = np.asarray(a["rates_all"], float)
        np.savez(new(L, d), rates=r, obj=np.asarray(a["obj_per_t"], float), Ks=Ks)
        print(f"  model {m}@{ep} -> test {L} δ={d:g}: obj {a['obj_per_t'].mean():.2f} "
              f"viol {100 * (r < d - 1e-9).mean():.3f}%  ({sum(sec) / 60:.1f} min)", flush=True)

    assert all(new(L, d).exists() for L, d in cells), "cells missing; not swapping"
    for L, d in cells:
        if cur(L, d).exists() and not arch(L, d).exists():
            shutil.move(cur(L, d), arch(L, d))
        shutil.move(new(L, d), cur(L, d))
    print(f"model {m}: epoch-{ep} cells promoted; previous cells archived as m{m}{args.archive_tag}_*", flush=True)
    print("averaged over the four test sizes:")
    summary(m, f"m{m}{args.archive_tag}")
    summary(m, f"m{m}")


if __name__ == "__main__":
    main()
