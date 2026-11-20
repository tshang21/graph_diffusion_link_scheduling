#!/usr/bin/env bash
# SAGNN on the 500-link irregular LCC dataset (100 train / 50 test graphs; the paper_plots2 base set).
#   [0] train the SAGNN policy on the train split (sagnn.collect_expert defaults: 100 epochs, lr_phi 5e-5,
#       lr_lambda 2, wd 0.05, 256 feats x 3 layers, delta 0.1). The policy is lambda-conditioned and
#       delta-agnostic: delta only enters the dual update at rollout time.
#   [1] per delta in DELTAS: dual-roll the policy on the 50 test graphs, T=100 slots from step 0 (no warmup
#       dropped, like the MWIS teacher), wd 0.05, no collision masking -- scored STRICTLY downstream
#       -> data/delta_grid/irregular_lcc_noise0.02/sagnn/test_T100_delta<d>/samples_SAGNN.pt
# Resumable: finished steps are skipped. Launch detached:
#   setsid bash scripts/run_sagnn_irregular500.sh > data/sagnn_irregular500.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"; export PYTHONPATH=.
DELTAS="${DELTAS:-0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1}"
TAG=irregular_lcc_noise0.02
DS="$ROOT/data/sawl_dataset_irregular_lcc_degree6_noise0.02_graphs_150_nodes256-324.pt"
BASE="$ROOT/data/delta_grid/irregular_lcc_noise0.02"
GRAPHS="$BASE/test_T100_delta0.1/samples_MWIS.pt"          # the 50 test graphs, same order as the test split
MODEL="models/exp_epochs_100_delta_0.1_lr_phi_5e-05_lr_lambda_2_weight_decay0.05_degree_6_of_1_${TAG}.pt"

if [ -f "$MODEL" ]; then
  echo "[0] reusing SAGNN policy $MODEL"
else
  python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "!! CUDA unavailable -- not training SAGNN on CPU"; exit 1; }
  echo "[0] training SAGNN on $DS ($(date '+%F %T'))"
  python -u -m sagnn.collect_expert --dataset "$DS" --tag "$TAG" --delta 0.1 --weight_decay 0.05 \
    || { echo "!! SAGNN training failed"; exit 1; }
  [ -f "$MODEL" ] || { echo "!! model not found at $MODEL"; exit 1; }
fi

for D in $DELTAS; do
  OUT="$BASE/sagnn/test_T100_delta${D}/samples_SAGNN.pt"
  [ -f "$OUT" ] && { echo "[1] reusing $OUT"; continue; }
  echo "[1] rollout delta=$D ($(date '+%T'))"
  python -u -m sagnn.collect_dataset --sagnn-model "$MODEL" --graphs "$GRAPHS" --out "$OUT" \
    --delta "$D" --weight-decay 0.05 --T 100 --warmup-steps 0 --seed 0 \
    || { echo "!! rollout failed delta=$D"; exit 1; }
  python - "$OUT" "$GRAPHS" "$D" <<'PYEOF'
import sys, numpy as np
import common.utils  # noqa: F401
from ugnn.dataset import load_samples
from ugnn.metrics import aggregate_method
roll, graphs, d = load_samples(sys.argv[1]), load_samples(sys.argv[2]), float(sys.argv[3])
a = aggregate_method([g["thresholded_schedule"] for g in roll], graphs, d)
r = np.asarray(a["rates_all"], float)
print(f"    strict: objective {np.mean(a['obj_per_t']):.2f} | {100 * (r < d - 1e-9).mean():.2f}% links below delta")
PYEOF
done
echo "SAGNN_IRREGULAR500_DONE $(date '+%F %T')"
