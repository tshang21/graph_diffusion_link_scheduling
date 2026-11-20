"""Has an extension checkpoint caught up with the 500- and 1000-link models?

    cd src && PYTHONPATH=. python ../scripts/early_stop_check.py --model 2000 --ckpt <path>

Rolls the checkpoint on all four test families at delta = 0.1 (cells cached in
results/_extension_cache as m<model>_ep<epoch>_on<L>_d0.1.npz, which transfer_row.py reuses) and
compares it with the 500- and 1000-link models already in the size-transfer matrix. Criterion
(fixed in advance): at EVERY test size, objective above both references and share of links below
delta below both. Exit code 0 = criterion met, 3 = not yet.
"""
import argparse
import sys
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
SIZES, REFS, DELTA = ("500", "1000", "2000", "5000"), ("500", "1000"), 0.1


def scores(f):
    z = np.load(f)
    return float(z["obj"].mean()), 100 * float((z["rates"] < DELTA - 1e-9).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="training size of the model: 2000|5000")
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()
    m = args.model
    ep = int(torch.load(args.ckpt, map_location="cpu", weights_only=False)["epoch"]) + 1
    print(f"=== check: model {m} @ epoch {ep}, δ={DELTA:g} -- must beat the 500- and 1000-link models at every size", flush=True)
    met = True
    for L in SIZES:
        f = EXT / f"m{m}_ep{ep}_on{L}_d{DELTA:g}.npz"
        if not f.exists():
            gs = load_samples(BASE / "test_T100_delta0.1/samples_MWIS.pt" if L == "500"
                              else ROOT / f"data/irregular_links{L}/graphs_test50.pt")
            sch, _ = rollout_ugnn(args.ckpt, gs, T=100, device="cuda", seed=0, delta=DELTA)
            a = aggregate_method(sch, gs, DELTA)
            np.savez(f, rates=np.asarray(a["rates_all"], float), obj=np.asarray(a["obj_per_t"], float),
                     Ks=np.array([graph_K(g) for g in gs]))
        o, v = scores(f)
        ref = {r: scores(TC / f"m{r}_on{L}_d{DELTA:g}.npz") for r in REFS}
        ok_o = o > max(x[0] for x in ref.values())
        ok_v = v < min(x[1] for x in ref.values())
        met = met and ok_o and ok_v
        print(f"  test {L:>4}:  obj {o:.2f} vs {ref['500'][0]:.2f}/{ref['1000'][0]:.2f} {'✓' if ok_o else '✗'}"
              f"   viol {v:.2f}% vs {ref['500'][1]:.2f}/{ref['1000'][1]:.2f}% {'✓' if ok_v else '✗'}", flush=True)
    print(f"=== model {m} @ epoch {ep}: {'CRITERION MET' if met else 'not yet'}", flush=True)
    sys.exit(0 if met else 3)


if __name__ == "__main__":
    main()
