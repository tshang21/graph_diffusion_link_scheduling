#!/bin/bash
# File-based MWIS pipeline: per graph, solve one weighted MIS per timestep with
# KaMIS, evolving node weights between slots via KaMIS.py (paced deficit rule:
# w = 1 + ETA*max(0, DELTA*(t+1) - times_scheduled) -> DELTA sets the target, ETA the priority).
# Finishes by stacking all solver outputs into samples_MWIS.pt (UGNN-trainable).
#
# All config is env-overridable, e.g.:
#   TIMESTEPS=5 GRAPH_START=0 GRAPH_END=0 OUT_DIR=/tmp/mwis_smoke ./run_pipeline.sh
set -euo pipefail

# Config
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
TIMESTEPS=${TIMESTEPS:-100}
GRAPH_START=${GRAPH_START:-0}
GRAPH_END=${GRAPH_END:-99}                 # inclusive
DATASET=${DATASET:-data/sawl_dataset_degree6_split150.pt}
SPLIT=${SPLIT:-test}
INIT_WEIGHT=${INIT_WEIGHT:-1}
OUT_DIR=${OUT_DIR:-data/mwis_pipeline}
KAMIS_DIR=${KAMIS_DIR:-$SCRIPT_DIR/external/KaMIS/deploy}
TIME_LIMIT=${TIME_LIMIT:-0.01}             # s/slot; weighted_local_search is anytime (default 1000s!)
SEED=${SEED:-0}
WORKERS=${WORKERS:-}                       # parallel graphs (empty -> KaMIS.py default: min(24, cores-2))
CLEAN=${CLEAN:-0}                          # 1 = delete per-graph .gr/weights after each trajectory
STORE_A=${STORE_A:-1}                      # 0 = schedules-only export (big-K: dense A would be tens of GB)
DELTA_MAX=${DELTA_MAX:-0}                   # >0: draw each graph's delta from U(0, DELTA_MAX) instead of DELTA
DELTA=${DELTA:-0.1}                        # minimum-rate target driving the weight updates (also in the export meta)
ETA=${ETA:-2}                              # dual step size eta (arXiv:2607.18480 uses 2)
WEIGHT_RULE=${WEIGHT_RULE:-dual}           # dual (dual dynamics) | deficit (the earlier rule)
WEIGHT_SCALE=${WEIGHT_SCALE:-100}          # integer discretization only (argmax-invariant)

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"
TIMES_FILE=$OUT_DIR/mwis_times.tmp
: > "$TIMES_FILE"

# All graphs in ONE python invocation: the dataset loads once, then graphs fan out
# across a spawn pool (WORKERS, default min(24, cores-2)); per graph the trajectory
# runs init -> (sort_adjacencies -> weighted_local_search -> evolve) x T in-process.
# Identical files to the original bash-driven loop (kept as init/evolve modes).
python "$SCRIPT_DIR/KaMIS.py" --mode run --graph_start $GRAPH_START --graph_end $GRAPH_END \
    --T $TIMESTEPS --out_dir "$OUT_DIR" --dataset "$DATASET" --split "$SPLIT" \
    --init_weight $INIT_WEIGHT --kamis_dir "$KAMIS_DIR" --time_limit $TIME_LIMIT --seed $SEED --delta $DELTA --eta $ETA --delta-max $DELTA_MAX \
    --weight-rule $WEIGHT_RULE --weight-scale $WEIGHT_SCALE \
    ${WORKERS:+--workers $WORKERS} $([ "$CLEAN" = "1" ] && echo --clean)

echo "Average MWIS runtime and standard deviation:"
awk '{
    n++;
    sum += $1;
    sumsq += $1^2
}
END {
    if (n > 1) {
        mean = sum / n;
        std = sqrt((sumsq - sum^2 / n) / (n - 1));
        printf "Mean: %.6f\nStd Dev: %.6f\n", mean, std
    } else if (n == 1) {
        printf "Mean: %.6f\nStd Dev: 0.000000\n", sum
    } else {
        print "No data"
    }
}' "$TIMES_FILE"

# Step 3: stack all mwis_*_t*.out into a UGNN-trainable samples_MWIS.pt
PYTHONPATH=src python "$SCRIPT_DIR/KaMIS.py" --mode export \
    --out_dir "$OUT_DIR" --dataset "$DATASET" --split "$SPLIT" --T $TIMESTEPS --delta $DELTA --eta $ETA \
    --weight-rule $WEIGHT_RULE --weight-scale $WEIGHT_SCALE \
    --init_weight $INIT_WEIGHT \
    $([ "$STORE_A" = "0" ] && echo --no-store-A)
