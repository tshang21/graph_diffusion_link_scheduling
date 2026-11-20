#!/usr/bin/env bash
# Extend the 2000- and 5000-link native models 200 epochs at a time and stop as soon as a checkpoint
# beats both the 500- and 1000-link models at delta = 0.1 on every test size -- objective above both,
# violation below both (scripts/early_stop_check.py). Capped at 1000 epochs. The chosen checkpoint is
# then swapped into the size-transfer matrix (scripts/transfer_row.py). Resumable: after any crash,
# reboot or driver upgrade, re-run the same command:
#   setsid bash scripts/run_extend_early_stop.sh > data/extend_early_stop.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"; export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEP="${STEP:-200}"; CAP="${CAP:-1000}"
EXT="$ROOT/data/delta_grid/irregular_lcc_noise0.02/results/_extension_cache"
declare -A BATCH=([2000]=13 [5000]=5) OLD_EPOCHS=([2000]=500 [5000]=300)   # batch*K held at 60*440
FILTER='UserWarning|warnings.warn|it/s\]|s/it\]|TimeEmbedding'

need_cuda() {   # a driver upgrade mid-run silently drops new processes to CPU
  python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
    || { echo "!! CUDA unavailable ($1) -- stopping; reboot after a driver upgrade, then re-run"; exit 1; }
}
epoch_of() {
  python -c "import sys, torch; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch']) + 1)" "$1"
}

for L in ${SIZES:-2000 5000}; do
  RUNS="$ROOT/data/irregular_links$L/runs"; OLD=${OLD_EPOCHS[$L]}
  CHOSEN="$EXT/L${L}_chosen_checkpoint.txt"
  if [ ! -f "$CHOSEN" ]; then
    CK=""; EP=0                                    # newest state: any _ext checkpoint, else the original
    for f in "$RUNS"/*_ugnn_native_L${L}_ext/*.pt; do
      [ -f "$f" ] || continue
      e=$(epoch_of "$f"); [ "$e" -gt "$EP" ] && { CK="$f"; EP=$e; }
    done
    [ -n "$CK" ] || { CK="$(python -m ugnn.model --name "ugnn_native_L$L" 2>/dev/null | tail -1)"; EP=$(epoch_of "$CK"); }
    while :; do
      if [ "$EP" -gt "$OLD" ] && [[ "$CK" != *_last.pt ]]; then      # a finished 200-epoch segment: check it
        need_cuda "L$L check at epoch $EP"
        python -u "$ROOT/scripts/early_stop_check.py" --model "$L" --ckpt "$CK" 2>&1 | grep -vE "$FILTER"
        rc=${PIPESTATUS[0]}
        if [ "$rc" -eq 0 ] || { [ "$rc" -eq 3 ] && [ "$EP" -ge "$CAP" ]; }; then
          echo "$CK" > "$CHOSEN"
          echo "=== L$L: using epoch $EP -- $([ "$rc" -eq 0 ] && echo 'criterion met' || echo "cap of $CAP epochs reached") ($(date +%H:%M))"
          break
        fi
        [ "$rc" -eq 3 ] || { echo "!! check failed for L=$L at epoch $EP"; exit 1; }
      fi
      TARGET=$(( OLD + ((EP - OLD) / STEP + 1) * STEP )); [ "$TARGET" -gt "$CAP" ] && TARGET=$CAP
      need_cuda "L$L training to epoch $TARGET"
      echo "=== L$L: training from $(basename "$CK") (epoch $EP) to epoch $TARGET ($(date +%H:%M))"
      python -u -m ugnn.train --data "$ROOT/data/irregular_links$L/train_T100_grid.pt" \
        --name "ugnn_native_L${L}_ext" --results-dir "$RUNS" --window 10 --stride 1 --windows 10 \
        --epochs "$TARGET" --batch-size "${BATCH[$L]}" --accum 2 --lr 1e-4 --resume "$CK" \
        --checkpoint-every 20 --keep-local-model \
        || { echo "!! training failed L=$L (re-run to resume)"; exit 1; }
      D="$(ls -td "$RUNS"/*_ugnn_native_L${L}_ext | head -1)"; CK="$D/$(basename "$D").pt"; EP=$TARGET
      rm -f "$RUNS"/*_ugnn_native_L${L}_ext/*_last.pt
      python -m wandb artifact cache cleanup 500MB > /dev/null 2>&1   # every segment stages a 318 MB upload
    done
  fi
  KEEP="$(cat "$CHOSEN")"
  for f in "$RUNS"/*_ugnn_native_L${L}_ext/*.pt; do [ "$f" = "$KEEP" ] || rm -f "$f"; done   # the rest are on W&B
  need_cuda "L$L matrix rollouts"
  python -u "$ROOT/scripts/transfer_row.py" --model "$L" --ckpt "$KEEP" --archive-tag "e$OLD" 2>&1 | grep -vE "$FILTER"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "!! matrix update failed L=$L"; exit 1; }
done
echo "EARLY_STOP_RUN_DONE $(date '+%F %T')"
