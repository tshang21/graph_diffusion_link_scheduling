#!/usr/bin/env bash
# Fine-tune the settled 500-link UGNN (bc512, 1000 epochs, delta in {0.01, 0.05, 0.1}) on the same training graphs
# with delta = 0.15 ADDED, for the row figure of paper_plots2 only (the transfer figure keeps the settled model).
#   [1] MWIS teacher on the 100 TRAIN graphs at delta = 0.15 (dual rule, eta 2, scale 1000, T = 100, 0.1 s/slot) -- CPU
#       -> data/delta_grid/irregular_lcc_noise0.02/train_T100_delta0.15/samples_MWIS.pt
#   [2] 4-delta training set = the settled model's train_T100_grid.pt + the delta = 0.15 trajectories
#       -> train_T100_grid_d015.pt   (the original grid file is left untouched)
#   [3] resume the settled checkpoint for FT_EPOCHS more epochs at the settled hyper-parameters (weights AND AdamW
#       moments are restored; --epochs is the TOTAL), snapshot every 100 epochs -> runs/<ts>_ugnn_w10_bc512_ft015/
#   [4] every snapshot vs the settled model on the 50 TEST graphs with the paper's sampler (10 DDIM steps, bf16):
#       has delta <= 0.1 moved, and what did delta = 0.15 gain?  (scripts/eval_finetune_delta015.py)
# ~2 h on the 3080 Ti for 300 epochs (4000 examples/epoch vs 3000 -> ~23.6 s/epoch). Resumable: [1]/[2] skip when
# their output exists; re-running [3] starts a NEW run directory. Launch detached:
#   setsid bash scripts/run_finetune_delta015.sh > data/finetune_delta015.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/envs/gsd/bin/python}"; export PATH="$(dirname "$PY"):$PATH"
BASE="$ROOT/data/delta_grid/irregular_lcc_noise0.02"
DS="$ROOT/data/sawl_dataset_irregular_lcc_degree6_noise0.02_graphs_150_nodes256-324.pt"
CKPT="$(ls "$BASE"/runs/*ugnn_w10_bc512/*_ugnn_w10_bc512.pt)"         # the settled 1000-epoch model
FT_EPOCHS="${FT_EPOCHS:-300}"; D=0.15
TEACH="$BASE/train_T100_delta$D"; GRID="$BASE/train_T100_grid_d015.pt"

if [ -f "$TEACH/samples_MWIS.pt" ]; then echo "[1] reusing $TEACH/samples_MWIS.pt"; else
  echo "[1] MWIS teacher, 100 train graphs, delta=$D ($(date '+%F %T'))"
  DATASET="$DS" SPLIT=train GRAPH_START=0 GRAPH_END=99 TIMESTEPS=100 INIT_WEIGHT=1 DELTA="$D" DELTA_MAX=0 ETA=2 \
    WEIGHT_RULE=dual WEIGHT_SCALE=1000 OUT_DIR="$TEACH" TIME_LIMIT=0.1 SEED=0 \
    bash "$ROOT/src/MIS/run_pipeline.sh" || { echo "!! teacher collection failed"; exit 1; }
  find "$TEACH" -maxdepth 1 \( -name 'graph_*.gr' -o -name 'weights_*.pt' -o -name 'edges_*.pt' \
      -o -name 'mwis_times.tmp' -o -name '*_sorted.gr' \) -delete
fi

cd "$ROOT/src"; export PYTHONPATH=.
if [ -f "$GRID" ]; then echo "[2] reusing $GRID"; else
  "$PY" - "$BASE" "$TEACH/samples_MWIS.pt" "$GRID" "$D" <<'PYEOF' || { echo "!! grid build failed"; exit 1; }
import sys
import common.utils  # noqa: F401
from ugnn.dataset import load_samples, save_samples
base, teach, out, d = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
grid = load_samples(f"{base}/train_T100_grid.pt")
new = [{"A": g["A"], "raw_schedule": g["raw_schedule"], "delta": d} for g in load_samples(teach)]
deltas = sorted({float(g["delta"]) for g in grid} | {d})
save_samples(out, grid + new, {"source": "MWIS dual", "deltas": deltas, "T": 100, "per_graph_delta": True,
                               "deconfounded": "every graph rolled at every delta"})
print(f"[2] {len(grid)} + {len(new)} trajectories, deltas {deltas} -> {out}")
PYEOF
fi

TOTAL=$((1000 + FT_EPOCHS))
echo "[3] fine-tuning $CKPT -> epoch $TOTAL ($(date '+%F %T'))"
"$PY" -u -m ugnn.train --data "$GRID" --name ugnn_w10_bc512_ft015 --results-dir "$BASE/runs" \
  --window 10 --stride 1 --windows 10 --epochs "$TOTAL" --batch-size 60 --accum 2 --lr 1e-4 --resume "$CKPT" \
  --checkpoint-every 20 --save-every-n-epochs 100 --keep-local-model || { echo "!! fine-tuning failed"; exit 1; }

echo "[4] snapshots vs the settled model ($(date '+%F %T'))"
RUN="$(ls -d "$BASE"/runs/*_ugnn_w10_bc512_ft015 | tail -1)"
"$PY" -u "$ROOT/scripts/eval_finetune_delta015.py" --settled "$CKPT" --run "$RUN" || { echo "!! evaluation failed"; exit 1; }
echo "FINETUNE_D015_DONE $(date '+%F %T')"
