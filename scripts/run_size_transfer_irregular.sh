#!/usr/bin/env bash
# Size-transfer on the irregular LCC distribution: does the settled 500-link UGNN transfer
# zero-shot to bigger graphs from the SAME distribution, and how does it compare to models
# trained natively at each size?
#
# The four families (data/irregular_links{1000,2000,5000} + the 500-link base) are matched in
# CONFLICT DEGREE (mean 5.74-5.87, max 14-15) -- absolute grid jitter is held fixed, so only the
# graph SIZE varies. Graph files store edge_index+K, not a dense A (100 MB/graph at K=5000).
#
#   [1] MWIS dual teacher per size x delta x split      (STORE_A=0: schedules only)
#   [2] merge schedules with edge_index -> <size>/train_T100_grid.pt for training
#   [3] train a native UGNN per size, settled config, batch scaled to hold batch*K constant
#   [4] 4 models x 4 test sets x 3 deltas of rollouts -> results/_transfer_cache
#
#   ./scripts/run_size_transfer_irregular.sh            # everything, resumable
#   SIZES=1000 STAGES="1 2 3" ./scripts/run_size_transfer_irregular.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"; export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-python}"

SIZES="${SIZES:-1000 2000 5000}"
DELTAS="${DELTAS:-0.01 0.05 0.1}"
T="${T:-100}"; SEED="${SEED:-0}"
# epoch budget scaled by size: bigger graphs carry far more links per example, and per-epoch
# cost grows ~linearly in K, so equal epochs would both over-train and over-spend at the top end.
EPOCHS_1000="${EPOCHS_1000:-700}"; EPOCHS_2000="${EPOCHS_2000:-500}"; EPOCHS_5000="${EPOCHS_5000:-300}"
BASE_CHANNELS="${BASE_CHANNELS:-512}"
# rolling checkpoint: a hang or reboot costs at most this many epochs, and a re-run resumes
CKPT_EVERY="${CKPT_EVERY:-10}"
WINDOW="${WINDOW:-10}"; WINDOWS="${WINDOWS:-10}"; LR="${LR:-1e-4}"
STAGES="${STAGES:-1 2 3 4}"
SPLITS="${SPLITS:-train test}"               # stage-1 teacher splits
BASE_DIR="$ROOT/data/delta_grid/irregular_lcc_noise0.02"
OUTD="$BASE_DIR/results/_transfer_cache"; mkdir -p "$OUTD"
has(){ [[ " $STAGES " == *" $1 "* ]]; }

# batch*K held at the settled run's 60*440; accum 2 keeps the effective batch matched
batch_for(){ $PY -c "print(max(4, round(60*440/$1)))"; }

for L in $SIZES; do
  GD="$ROOT/data/irregular_links${L}"
  KMED=$($PY -c "
import torch,sys; sys.path.insert(0,'.')
import common.utils
from ugnn.dataset import load_samples, graph_K
import numpy as np
print(int(np.median([graph_K(g) for g in load_samples('$GD/graphs_train100.pt')])))")
  BS=$(batch_for "$KMED")
  echo "=== links=$L  K median $KMED  batch $BS accum 2"

  if has 1; then
    for SPLIT in $SPLITS; do
      N=$([ "$SPLIT" = train ] && echo 100 || echo 50)
      for D in $DELTAS; do
        OUT="$GD/${SPLIT}_T${T}_delta${D}"
        [ -f "$OUT/samples_MWIS.pt" ] && { echo "  reuse $SPLIT δ=$D"; continue; }
        echo "  [1] teacher $SPLIT δ=$D ($(date +%H:%M:%S))"
        DATASET="$GD/sawl_split100.pt" SPLIT="$SPLIT" GRAPH_START=0 GRAPH_END=$((N-1)) \
          TIMESTEPS="$T" INIT_WEIGHT=1 DELTA="$D" DELTA_MAX=0 ETA=2 WEIGHT_RULE=dual \
          WEIGHT_SCALE=1000 STORE_A=0 OUT_DIR="${OUT#$ROOT/}" TIME_LIMIT=0.1 SEED="$SEED" \
          bash "$ROOT/src/MIS/run_pipeline.sh" || { echo "  !! teacher failed L=$L $SPLIT δ=$D"; exit 1; }
        find "$OUT" -maxdepth 1 \( -name 'graph_*.gr' -o -name 'weights_*.pt' -o -name 'edges_*.pt' \
          -o -name 'mwis_times.tmp' -o -name '*_sorted.gr' -o -name 'mwis_*.out' \) -delete
      done
    done
  fi

  if has 2 && [ ! -f "$GD/train_T100_grid.pt" ]; then
    echo "  [2] merging schedules with edge_index -> train_T100_grid.pt"
    $PY - "$GD" "$T" $DELTAS <<'PYEOF'
import sys
from ugnn.dataset import load_samples, save_samples, graph_edges, graph_K
gd, T, deltas = sys.argv[1], sys.argv[2], sys.argv[3:]
graphs = load_samples(f"{gd}/graphs_train100.pt")          # edge_index + K
out = []
for d in deltas:
    s = load_samples(f"{gd}/train_T{T}_delta{d}/samples_MWIS.pt")
    assert len(s) == len(graphs), f"{len(s)} samples vs {len(graphs)} graphs"
    for g, x in zip(graphs, s):
        assert graph_K(g) == int(x["K"]), "graph/sample K mismatch"
        out.append({"edge_index": graph_edges(g), "K": graph_K(g),
                    "raw_schedule": x["raw_schedule"], "delta": float(d)})
save_samples(f"{gd}/train_T{T}_grid.pt", out,
             {"source": "MWIS dual", "deltas": [float(x) for x in deltas], "T": int(T),
              "per_graph_delta": True, "sparse": True})
print(f"    merged {len(out)} trajectories -> {gd}/train_T{T}_grid.pt")
PYEOF
  fi

  if has 3; then
    # train.py falls back to CPU without complaint when CUDA is unavailable (e.g. an unattended NVIDIA
    # driver upgrade leaves the loaded kernel module older than the libraries) -- never train like that
    $PY -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
      || { echo "  !! CUDA unavailable -- refusing to train L=$L on CPU (reboot after a driver upgrade)"; exit 1; }
    if $PY -m ugnn.model --name "ugnn_native_L${L}" >/dev/null 2>&1; then
      echo "  [3] native model already on W&B, skipping"
    else
      EP_VAR="EPOCHS_${L}"; EPOCHS="${!EP_VAR}"
      LAST="$(ls -t "$GD"/runs/*_ugnn_native_L${L}/*_last.pt 2>/dev/null | head -1)"
      RESUME=""; [ -n "$LAST" ] && RESUME="--resume $LAST"
      echo "  [3] training native bc${BASE_CHANNELS}, $EPOCHS epochs${LAST:+, resuming $LAST} ($(date +%H:%M:%S))"
      $PY -u -m ugnn.train --data "$GD/train_T${T}_grid.pt" --name "ugnn_native_L${L}" \
        --results-dir "$GD/runs" --window "$WINDOW" --stride 1 --windows "$WINDOWS" \
        --epochs "$EPOCHS" --batch-size "$BS" --accum 2 --lr "$LR" \
        --base-channels "$BASE_CHANNELS" --gnn-layers 2 --gnn-k 2 \
        --checkpoint-every "$CKPT_EVERY" $RESUME \
        && { [ -z "$LAST" ] || rm -f "$LAST"; } \
        || { echo "  !! native training failed L=$L (re-run to resume from its _last.pt)"; exit 1; }
    fi
  fi
done

if has 4; then
  $PY -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
    || { echo "!! CUDA unavailable -- refusing to run rollouts"; exit 1; }
  echo "=== [4] transfer matrix: 4 models x 4 test sets x ${DELTAS// /,} ($(date +%H:%M:%S))"
  $PY - "$OUTD" "$ROOT" $DELTAS <<'PYEOF'
import sys
from pathlib import Path

import numpy as np

import common.utils  # noqa: F401  SAWLData alias
from ugnn.dataset import graph_K, load_samples
from ugnn.generate import rollout_ugnn
from ugnn.metrics import aggregate_method
from ugnn.model import fetch_model

outd, root, deltas = Path(sys.argv[1]), Path(sys.argv[2]), [float(x) for x in sys.argv[3:]]
BASE = root / "data/delta_grid/irregular_lcc_noise0.02"
SIZES = ("500", "1000", "2000", "5000")
_ck = {}


def ckpt(m):
    """Checkpoint for the model trained on m links, or None while it is still untrained.

    Resolved lazily, so cached cells never download a model and an untrained size is skipped
    (its rollouts run on a later pass, once it exists)."""
    if m not in _ck:
        _ck[m] = (str(next((BASE / "runs").glob("*ugnn_w10_bc512/*_epoch700.pt"))) if m == "500"
                  else fetch_model(f"ugnn_native_L{m}"))
        if _ck[m] is None:
            print(f"    model {m}: not trained yet, skipping its rollouts", flush=True)
    return _ck[m]


def testset(name):
    if name == "500":
        return load_samples(BASE / "test_T100_delta0.1/samples_MWIS.pt")
    return load_samples(root / f"data/irregular_links{name}/graphs_test50.pt")


for tname in SIZES:
    gs = None
    for mname in SIZES:
        for d in deltas:
            f = outd / f"m{mname}_on{tname}_d{d:g}.npz"
            if f.exists() or ckpt(mname) is None:
                continue
            gs = gs or testset(tname)
            sch, sec = rollout_ugnn(ckpt(mname), gs, T=100, device="cuda", seed=0, delta=d)
            a = aggregate_method(sch, gs, d)
            r = np.asarray(a["rates_all"], float)
            np.savez(f, rates=r, obj=np.asarray(a["obj_per_t"], float),
                     Ks=np.array([graph_K(g) for g in gs]))
            print(f"    model {mname} -> test {tname} δ={d:g}: obj {a['obj_per_t'].mean():.2f} "
                  f"viol {100 * (r < d - 1e-9).mean():.3f}%  ({sum(sec) / 60:.1f} min)", flush=True)
PYEOF
fi
echo "TRANSFER_DONE $(date +%H:%M:%S)"
