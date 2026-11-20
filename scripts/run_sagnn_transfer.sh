#!/usr/bin/env bash
# The 500-link SAGNN policy (scripts/run_sagnn_irregular500.sh) dual-rolled zero-shot on the larger irregular
# test families, so Fig 8 (paper_plots2) has a SAGNN value at every test size:
#   per size in SIZES, per delta in DELTAS: 50 test graphs, T=100 from lambda=0 (no warmup dropped), wd 0.05,
#   no collision masking -- scored STRICTLY downstream
#   -> data/irregular_links<L>/sagnn/test_T100_delta<d>/samples_SAGNN.pt   (same layout as the 500-link set)
# CPU only (one core per graph in the spawn pool). Resumable: finished files are skipped. Launch detached:
#   setsid bash scripts/run_sagnn_transfer.sh > data/sagnn_transfer.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"; export PYTHONPATH=.
SIZES="${SIZES:-1000 2000 5000}"
DELTAS="${DELTAS:-0.1 0.01 0.05}"                             # the figure's delta first
MODEL=models/exp_epochs_100_delta_0.1_lr_phi_5e-05_lr_lambda_2_weight_decay0.05_degree_6_of_1_irregular_lcc_noise0.02.pt
[ -f "$MODEL" ] || { echo "!! SAGNN policy not found: $MODEL"; exit 1; }

for D in $DELTAS; do
  for L in $SIZES; do
    GRAPHS="$ROOT/data/irregular_links${L}/graphs_test50.pt"
    OUT="$ROOT/data/irregular_links${L}/sagnn/test_T100_delta${D}/samples_SAGNN.pt"
    [ -f "$OUT" ] && { echo "[skip] $OUT"; continue; }
    echo "[roll] links=$L delta=$D ($(date '+%F %T'))"
    python -u -m sagnn.collect_dataset --sagnn-model "$MODEL" --graphs "$GRAPHS" --out "$OUT" \
      --delta "$D" --weight-decay 0.05 --T 100 --warmup-steps 0 --seed 0 \
      || { echo "!! rollout failed links=$L delta=$D"; exit 1; }
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
done
echo "SAGNN_TRANSFER_DONE $(date '+%F %T')"
