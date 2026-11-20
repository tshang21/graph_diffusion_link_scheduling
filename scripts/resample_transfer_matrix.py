"""Re-roll the whole size-transfer matrix (4 models x 4 test families x 3 deltas) with another sampler setting.

    cd src && PYTHONPATH=. python ../scripts/resample_transfer_matrix.py --sampling-timesteps 10 --bf16

Same checkpoints and protocol as the canonical matrix in results/_transfer_cache (scripts/transfer_row.py): T=100,
seed 0, strict scoring. Cells go to results/_transfer_cache_<tag>/m<model>_on<L>_d<delta>.npz with
tag = s<steps>[bf16], so the canonical 50-step matrix is left untouched. Before anything is rolled, every checkpoint
is checked against its canonical m<model>_on500_d0.1 cell at the default sampler. Finished cells are skipped.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import common.utils  # noqa: F401  SAWLData alias
from ugnn.dataset import graph_K, load_samples
from ugnn.generate import rollout_ugnn
from ugnn.metrics import aggregate_method

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/delta_grid/irregular_lcc_noise0.02"
TC = BASE / "results/_transfer_cache"
SIZES, DELTAS = ("500", "1000", "2000", "5000"), (0.1, 0.05, 0.01)
EPOCHS = {"500": 1000, "1000": 1000, "2000": 1000, "5000": 500}     # the epochs of the canonical matrix rows


def ckpt(m):
    """The checkpoint behind the canonical m<m>_* row: the latest one saved at EPOCHS[m]."""
    runs, pat = (BASE / "runs", "*ugnn_w10_bc512/*.pt") if m == "500" else (ROOT / f"data/irregular_links{m}/runs", "*ugnn_native_L*/*.pt")
    return str([f for f in sorted(runs.glob(pat))
                if torch.load(f, map_location="cpu", weights_only=False)["epoch"] == EPOCHS[m] - 1][-1])


def testset(L):
    if L == "500":
        return load_samples(BASE / "test_T100_delta0.1/samples_MWIS.pt")
    return load_samples(ROOT / f"data/irregular_links{L}/graphs_test50.pt")


def roll(ck, gs, d, **sampler):
    sch, sec = rollout_ugnn(ck, gs, T=100, device="cuda", seed=0, delta=d, **sampler)
    a = aggregate_method(sch, gs, d)
    return np.asarray(a["rates_all"], float), np.asarray(a["obj_per_t"], float), sum(sec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sampling-timesteps", type=int, required=True, help="DDIM steps per sample call")
    p.add_argument("--bf16", action="store_true", help="sample under bfloat16 autocast")
    p.add_argument("--dry-run", action="store_true", help="print the checkpoints and the plan, roll nothing")
    args = p.parse_args()
    out = BASE / f"results/_transfer_cache_s{args.sampling_timesteps}{'bf16' if args.bf16 else ''}"
    cks = {m: ckpt(m) for m in SIZES}
    for m, c in cks.items():
        print(f"model {m:>4} @ epoch {EPOCHS[m]}: {c}", flush=True)
    todo = [(m, L, d) for d in DELTAS for m in SIZES for L in SIZES if not (out / f"m{m}_on{L}_d{d:g}.npz").exists()]
    print(f"{len(todo)} cells to roll -> {out}", flush=True)
    if args.dry_run:
        return

    graphs = {L: testset(L) for L in SIZES}
    for m in sorted({m for m, _, _ in todo}, key=int):              # right checkpoint? reproduce its canonical 500-link cell
        _, obj, _ = roll(cks[m], graphs["500"], 0.1)
        ref = float(np.load(TC / f"m{m}_on500_d0.1.npz")["obj"].mean())
        print(f"check model {m}: objective {obj.mean():.3f} vs canonical {ref:.3f}", flush=True)
        assert abs(obj.mean() - ref) < 0.05, f"model {m}: checkpoint does not reproduce the canonical matrix"

    out.mkdir(parents=True, exist_ok=True)
    for m, L, d in todo:
        r, obj, sec = roll(cks[m], graphs[L], d, sampling_timesteps=args.sampling_timesteps, bf16=args.bf16)
        np.savez(out / f"m{m}_on{L}_d{d:g}.npz", rates=r, obj=obj, Ks=np.array([graph_K(g) for g in graphs[L]]))
        print(f"  model {m} -> test {L} δ={d:g}: obj {obj.mean():.2f} viol {100 * (r < d - 1e-9).mean():.3f}%  ({sec:.0f} s)", flush=True)
    print("RESAMPLE_DONE", flush=True)


if __name__ == "__main__":
    main()
