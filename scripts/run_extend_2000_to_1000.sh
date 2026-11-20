#!/usr/bin/env bash
# Extend the 2000-link native model from its chosen epoch-700 checkpoint to 1000 epochs, then swap it
# into the size-transfer matrix (scripts/transfer_row.py; the epoch-700 cells are archived as m2000e700_*).
# Resumable: after any crash, reboot or driver upgrade, re-run the same command:
#   setsid bash scripts/run_extend_2000_to_1000.sh > data/extend_2000_to_1000.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"; export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=2000; TARGET=1000; RUNS="$ROOT/data/irregular_links$L/runs"
FILTER='UserWarning|warnings.warn|it/s\]|s/it\]|TimeEmbedding'

need_cuda() {   # a driver upgrade mid-run silently drops new processes to CPU
  python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
    || { echo "!! CUDA unavailable ($1) -- stopping; reboot after a driver upgrade, then re-run"; exit 1; }
}
epoch_of() {
  python -c "import sys, torch; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch']) + 1)" "$1"
}

CK="$(cat "$ROOT/data/delta_grid/irregular_lcc_noise0.02/results/_extension_cache/L${L}_chosen_checkpoint.txt")"
EP=$(epoch_of "$CK")
for f in "$RUNS"/*_ugnn_native_L${L}_ext/*.pt; do             # newest state; a final save beats a _last tie
  [ -f "$f" ] || continue
  e=$(epoch_of "$f")
  if [ "$e" -gt "$EP" ] || { [ "$e" -eq "$EP" ] && [[ "$f" != *_last.pt ]]; }; then CK="$f"; EP=$e; fi
done

if [ "$EP" -lt "$TARGET" ] || [[ "$CK" == *_last.pt ]]; then
  need_cuda "training to epoch $TARGET"
  echo "=== L$L: training from $(basename "$CK") (epoch $EP) to epoch $TARGET ($(date '+%F %T'))"
  python -u -m ugnn.train --data "$ROOT/data/irregular_links$L/train_T100_grid.pt" \
    --name "ugnn_native_L${L}_ext" --results-dir "$RUNS" --window 10 --stride 1 --windows 10 \
    --epochs "$TARGET" --batch-size 13 --accum 2 --lr 1e-4 --resume "$CK" \
    --checkpoint-every 20 --keep-local-model \
    || { echo "!! training failed (re-run to resume)"; exit 1; }
  D="$(ls -td "$RUNS"/*_ugnn_native_L${L}_ext | head -1)"; CK="$D/$(basename "$D").pt"
  rm -f "$RUNS"/*_ugnn_native_L${L}_ext/*_last.pt
  python -m wandb artifact cache cleanup 500MB > /dev/null 2>&1
fi
echo "=== L$L: epoch $(epoch_of "$CK") checkpoint $CK ($(date '+%F %T'))"

need_cuda "matrix rollouts"
python -u "$ROOT/scripts/transfer_row.py" --model "$L" --ckpt "$CK" --archive-tag e700 2>&1 | grep -vE "$FILTER"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "!! matrix update failed"; exit 1; }
echo "EXTEND_2000_DONE $(date '+%F %T')"
