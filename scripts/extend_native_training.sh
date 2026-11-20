#!/usr/bin/env bash
# Bring native size-transfer models up to TARGET_EPOCHS (default 1000, matching the 500-link model).
# Resumes each from its final W&B checkpoint (optimizer state included),
# keeps a checkpoint every 100 epochs, then rolls every new checkpoint on the model's own test size
# at EVAL_DELTAS. The original checkpoints' numbers already sit in the transfer cache
# (m<L>_on<L>_d<delta>.npz); results here go to results/_extension_cache, so the 4x4 matrix is untouched.
#
#   SIZES="2000 5000" ./scripts/extend_native_training.sh    # both to 1000 epochs
#   SIZES=5000 EVAL_DELTAS=0.1 ./scripts/extend_native_training.sh
# Re-running is safe: a size whose _ext final checkpoint exists is not retrained, and an interrupted
# run resumes from its _last.pt.
#   SMOKE=1 SIZES=1000 ./scripts/extend_native_training.sh  # evaluation path only: original ckpt, 5 graphs
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"; export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-python}"

SIZES="${SIZES:-2000 5000}"
TARGET_EPOCHS="${TARGET_EPOCHS:-1000}"
EVAL_DELTAS="${EVAL_DELTAS:-0.05 0.1}"
SMOKE="${SMOKE:-0}"
OUTD="$ROOT/data/delta_grid/irregular_lcc_noise0.02/results/_extension_cache"; mkdir -p "$OUTD"
declare -A BATCH=([1000]=26 [2000]=13 [5000]=5)       # batch*K held at the settled run's 60*440

need_cuda() {   # a driver upgrade mid-run silently drops new processes to CPU; check before each stage
  $PY -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
    || { echo "!! CUDA unavailable ($1) -- refusing to run"; exit 1; }
}
need_cuda "start"

evaluate() {   # $1 = size, remaining args = checkpoint paths
  local L="$1"; shift
  $PY - "$OUTD" "$ROOT" "$L" "$SMOKE" "$EVAL_DELTAS" "$@" <<'PYEOF'
import sys
from pathlib import Path

import numpy as np
import torch

import common.utils  # noqa: F401  SAWLData alias
from ugnn.dataset import load_samples
from ugnn.generate import rollout_ugnn
from ugnn.metrics import aggregate_method

outd, root, L, smoke = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4] == "1"
deltas = [float(x) for x in sys.argv[5].split()]
gs = load_samples(root / f"data/irregular_links{L}/graphs_test50.pt")[: 5 if smoke else None]
for ck in sys.argv[6:]:
    ep = int(torch.load(ck, map_location="cpu", weights_only=False)["epoch"]) + 1
    for d in deltas:
        f = outd / f"m{L}_ep{ep}_on{L}_d{d:g}{'_smoke' if smoke else ''}.npz"
        if f.exists():
            continue
        sch, sec = rollout_ugnn(ck, gs, T=100, device="cuda", seed=0, delta=d)
        a = aggregate_method(sch, gs, d)
        r = np.asarray(a["rates_all"], float)
        np.savez(f, rates=r, obj=np.asarray(a["obj_per_t"], float))
        print(f"    L{L} epoch {ep} δ={d:g}: obj {a['obj_per_t'].mean():.2f} "
              f"viol {100 * (r < d - 1e-9).mean():.3f}%  ({sum(sec) / 60:.1f} min)", flush=True)
PYEOF
}

for L in $SIZES; do
  ORIG="$($PY -m ugnn.model --name "ugnn_native_L${L}" 2>/dev/null | tail -1)"
  [ -f "$ORIG" ] || { echo "!! no original L$L checkpoint on W&B"; exit 1; }
  if [ "$SMOKE" = 1 ]; then evaluate "$L" "$ORIG" || exit 1; continue; fi

  RUNS="$ROOT/data/irregular_links${L}/runs"
  FINAL="$(ls "$RUNS"/*_ugnn_native_L${L}_ext/*_ugnn_native_L${L}_ext.pt 2>/dev/null | head -1)"
  if [ -n "$FINAL" ]; then
    echo "=== L$L: already extended ($(basename "$FINAL")), not retraining"
  else
    need_cuda "L$L training"
    LAST="$(ls -t "$RUNS"/*_ugnn_native_L${L}_ext/*_last.pt 2>/dev/null | head -1)"
    RESUME="${LAST:-$ORIG}"
    echo "=== L$L: resuming $(basename "$RESUME") -> $TARGET_EPOCHS epochs total ($(date +%H:%M:%S))"
    $PY -u -m ugnn.train --data "$ROOT/data/irregular_links${L}/train_T100_grid.pt" \
      --name "ugnn_native_L${L}_ext" --results-dir "$RUNS" --window 10 --stride 1 --windows 10 \
      --epochs "$TARGET_EPOCHS" --batch-size "${BATCH[$L]}" --accum 2 --lr 1e-4 --resume "$RESUME" \
      --checkpoint-every 20 --save-every-n-epochs 100 --keep-local-model \
      || { echo "!! extension training failed L=$L (re-run to resume from its _last.pt)"; exit 1; }
    rm -f "$RUNS"/*_ugnn_native_L${L}_ext/*_last.pt
  fi
  need_cuda "L$L evaluation"
  echo "=== L$L: evaluating the extension checkpoints ($(date +%H:%M:%S))"
  if evaluate "$L" $(ls "$RUNS"/*_ugnn_native_L${L}_ext/*.pt); then
    rm -f "$RUNS"/*_ugnn_native_L${L}_ext/*_ext_epoch*.pt   # scored; 318 MB each, the final checkpoint stays
  else
    echo "!! evaluation failed L=$L (checkpoints kept under $RUNS)"; exit 1
  fi
done
echo "EXTENSION_DONE $(date +%H:%M:%S)"
