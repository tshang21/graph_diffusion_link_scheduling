#!/usr/bin/env bash
# Training set for a delta-CONDITIONED UGNN, with delta DECONFOUNDED from the graph:
# every graph is rolled at EVERY delta in DELTAS, so the model sees the same A with several
# different targets and delta is the only thing that explains the difference.  (Rolling one
# random delta per graph instead makes delta a deterministic function of the graph, which the
# model can and does ignore.)  Trajectories are short: the dual dynamics decorrelate in ~5 slots
# and the per-link rate converges by ~40, so T=50 keeps the informative transient and drops the
# redundant stationary tail.  T must be >= 1/min(DELTAS) or the smallest delta is unrepresentable.
#
#   ./scripts/collect_delta_grid.sh                       # irregular, 100 graphs x 4 deltas x T=50
#   DELTAS="0.05 0.1" T=30 ./scripts/collect_delta_grid.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"; export PYTHONPATH=.
PY="${PY:-python}"

IRREGULAR="${IRREGULAR:-1}"
SIZE="${SIZE:-200-300}"; DEG="${DEG:-6}"
NOISE="${NOISE:-0.01}"; IRR_NODES="${IRR_NODES:-256-324}"; IRR_GRAPHS="${IRR_GRAPHS:-150}"
SPLIT="${SPLIT:-train}"; N_GRAPHS="${N_GRAPHS:-100}"
DELTAS="${DELTAS:-0.05 0.1 0.15 0.2}"
T="${T:-50}"
ETA="${ETA:-2}"; WEIGHT_RULE="${WEIGHT_RULE:-dual}"; WEIGHT_SCALE="${WEIGHT_SCALE:-1000}"
INIT_WEIGHT="${INIT_WEIGHT:-1}"; TIME_LIMIT="${TIME_LIMIT:-0.1}"; SEED="${SEED:-0}"; WORKERS="${WORKERS:-}"

if [ "$IRREGULAR" = "1" ]; then
    TAG="irregular_noise${NOISE}_nodes${IRR_NODES}"
    DS="${DS:-$ROOT/data/sawl_dataset_irregular_degree${DEG}_noise${NOISE}_graphs_${IRR_GRAPHS}_nodes${IRR_NODES}.pt}"
else
    TAG="$SIZE"; DS="${DS:-$ROOT/data/sawl_dataset_degree${DEG}_graphs_150_nodes${SIZE}.pt}"
fi
OUT_ROOT="${OUT_ROOT:-$ROOT/data/delta_grid/$TAG}"
[ -f "$DS" ] || { echo "ERROR: dataset missing: $DS" >&2; exit 1; }
echo "delta grid: $N_GRAPHS $SPLIT graphs x deltas [$DELTAS] x T=$T -> $OUT_ROOT"

for D in $DELTAS; do
    OUT="$OUT_ROOT/${SPLIT}_T${T}_delta${D}"
    if [ -f "$OUT/samples_MWIS.pt" ]; then echo "  reusing $OUT/samples_MWIS.pt"; continue; fi
    echo "  delta=$D ..."
    DATASET="$DS" SPLIT="$SPLIT" GRAPH_START=0 GRAPH_END=$((N_GRAPHS - 1)) TIMESTEPS="$T" \
        INIT_WEIGHT="$INIT_WEIGHT" DELTA="$D" DELTA_MAX=0 ETA="$ETA" \
        WEIGHT_RULE="$WEIGHT_RULE" WEIGHT_SCALE="$WEIGHT_SCALE" OUT_DIR="${OUT#$ROOT/}" \
        TIME_LIMIT="$TIME_LIMIT" SEED="$SEED" WORKERS="$WORKERS" \
        bash "$ROOT/src/MIS/run_pipeline.sh" || { echo "!! failed at delta=$D" >&2; exit 1; }
    find "$OUT" -maxdepth 1 \( -name 'graph_*.gr' -o -name 'weights_*.pt' -o -name 'edges_*.pt' \
        -o -name 'mwis_times.tmp' -o -name '*_sorted.gr' \) -delete
done

# concatenate the per-delta files into ONE training set: 4 entries per graph, same A, different delta
$PY - "$OUT_ROOT" "$SPLIT" "$T" $DELTAS <<'PYEOF'
import sys
from ugnn.dataset import load_samples, save_samples
root, split, T, deltas = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
out = []
for d in deltas:
    for g in load_samples(f"{root}/{split}_T{T}_delta{d}/samples_MWIS.pt"):
        out.append({"A": g["A"], "raw_schedule": g["raw_schedule"], "delta": float(d)})
path = f"{root}/{split}_T{T}_grid.pt"
save_samples(path, out, {"source": "MWIS dual", "deltas": [float(x) for x in deltas], "T": int(T),
                         "per_graph_delta": True, "deconfounded": "every graph rolled at every delta"})
n_g = len(out) // len(deltas)
print(f"grid: {n_g} graphs x {len(deltas)} deltas = {len(out)} trajectories x {T} slots -> {path}")
PYEOF
