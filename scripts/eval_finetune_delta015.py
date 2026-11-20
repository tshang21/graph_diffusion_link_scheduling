"""Did fine-tuning on delta = 0.15 move the settled model at delta <= 0.1, and what did delta = 0.15 gain?

    cd src && PYTHONPATH=. python ../scripts/eval_finetune_delta015.py --settled <settled.pt> --run <runs/..._ft015>

Rolls the settled checkpoint and every snapshot of the fine-tuning run (<exp>_epoch<N>.pt and the final <exp>.pt) on
the 50 test graphs at delta in {0.01, 0.05, 0.1, 0.12, 0.15, 0.2} with the paper's sampler (10 DDIM steps, bf16), T = 100,
seed 0, strict scoring, next to the MWIS teacher. Prints the table and writes it to <run>/eval_vs_settled.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import common.utils  # noqa: F401  SAWLData alias
from ugnn.dataset import load_samples
from ugnn.generate import rollout_ugnn
from ugnn.metrics import aggregate_method

BASE = Path(__file__).resolve().parents[1] / "data/delta_grid/irregular_lcc_noise0.02"
DELTAS = (0.01, 0.05, 0.1, 0.12, 0.15, 0.2)
SAMPLER = dict(sampling_timesteps=10, bf16=True)


def score(ckpt, graphs, d):
    sch, _ = rollout_ugnn(str(ckpt), graphs, T=100, device="cuda", seed=0, delta=d, **SAMPLER)
    a = aggregate_method(sch, graphs, d)
    r = np.asarray(a["rates_all"], float)
    return float(np.mean(a["obj_per_t"])), 100 * float((r < d - 1e-9).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--settled", required=True, help="the settled 1000-epoch checkpoint")
    p.add_argument("--run", required=True, help="run directory of the fine-tuning")
    args = p.parse_args()
    run = Path(args.run)
    snaps = sorted(run.glob("*_epoch*.pt"), key=lambda f: int(f.stem.split("_epoch")[1])) + [run / f"{run.name}.pt"]
    models = {"settled": Path(args.settled),
              **{("ft final" if f.stem == run.name else f"ft {f.stem.split('_epoch')[1]}"): f for f in snaps}}
    graphs = load_samples(BASE / "test_T100_delta0.1" / "samples_MWIS.pt")
    table = {}
    for d in DELTAS:
        rt = np.concatenate([np.asarray(g["raw_schedule"], np.float32).mean(0)
                             for g in load_samples(BASE / f"test_T100_delta{d:g}" / "samples_MWIS.pt")])
        table[f"{d:g}"] = {"MWIS": (100 * float(rt.mean()), 100 * float((rt < d - 1e-9).mean()))}
        for name, ck in models.items():
            table[f"{d:g}"][name] = score(ck, graphs, d)
    names = ["MWIS", *models]
    print(f"{'delta':>6} " + "".join(f"{n:>22}" for n in names) + "      (objective | % links below delta)")
    for d, row in table.items():
        print(f"{d:>6} " + "".join(f"{row[n][0]:>12.2f} |{row[n][1]:>7.2f}" for n in names))
    json.dump(table, open(run / "eval_vs_settled.json", "w"), indent=1)


if __name__ == "__main__":
    main()
