#!/bin/bash
#
# Reproduce the FiQA comparison for BAAI/bge-large-en-v1.5.
#
# Compares five conditions and reports Recall@20 and NDCG@10 for each:
#   1. pretrain    the base model with no finetuning
#   2. nopruning   standard finetuning on the full training set
#   3. sp          static (offline) pruning, kept_pct=0.5
#   4. infobatch   InfoBatch online pruning baseline
#   5. dp          dynamic pruning (OPERA)
#
# Usage:
#   ./reproduce_fiqa_bge.sh
#
# Environment variables:
#   OPERA_RAW_DATA_DIR  directory for raw and preprocessed data (required)
#   OPERA_WORK_ROOT     directory for checkpoints and results (required)
#   NUM_GPUS            number of GPUs for distributed finetuning (default: 8)
#   MAX_STEPS           training steps for the finetuned conditions (default: 2000)
#   BATCH_SIZE          per-device train batch size (default: 16)

set -e

# FiQA settings, matching the paper configuration.
TRAIN_DATA="beir/fiqa/train"
VAL_DATA="beir/fiqa/dev"
TEST_DATA="beir/fiqa/test"
MINING_MODE="hard"
MINING_RANGE="10-100"
KEPT_PCT="0.5"

MODEL_NAME="bge_large"
CONFIG_PREFIX="opera_bge_large"

: "${OPERA_RAW_DATA_DIR:?Set OPERA_RAW_DATA_DIR to the directory for raw and preprocessed data}"
: "${OPERA_WORK_ROOT:?Set OPERA_WORK_ROOT to the directory for checkpoints and results}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_STEPS="${MAX_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# A shared run id keeps the five experiment directories grouped together.
# Passed in so the script itself stays free of nondeterministic calls.
RUN_ID="${RUN_ID:-fiqa_bge}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_PARAMS="--train_data $TRAIN_DATA --val_data $VAL_DATA --test_data $TEST_DATA"

# Build per-run configs with the FiQA overrides applied.
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# The pretrain condition reuses the nopruning config but evaluates the base
# model instead of a checkpoint.
cp "configs/${CONFIG_PREFIX}_nopruning.yaml" "${TEMP_DIR}/${CONFIG_PREFIX}_pretrain.yaml"
sed -i 's/- "finetune"/- "pretrain"/' "${TEMP_DIR}/${CONFIG_PREFIX}_pretrain.yaml"

for cfg in nopruning sp infobatch dp; do
  cp "configs/${CONFIG_PREFIX}_${cfg}.yaml" "${TEMP_DIR}/${CONFIG_PREFIX}_${cfg}.yaml"
done

# Mining settings for the finetuned conditions. range_for_sampling and kept_pct
# have unique keys, so these substitutions do not touch offline_pruning.mode.
for cfg in nopruning sp infobatch dp; do
  f="${TEMP_DIR}/${CONFIG_PREFIX}_${cfg}.yaml"
  sed -i "s/range_for_sampling: \".*\"/range_for_sampling: \"${MINING_RANGE}\"/" "$f"
  sed -i "s/per_device_train_batch_size: .*/per_device_train_batch_size: ${BATCH_SIZE}/" "$f"
done

# Only static pruning removes training pairs offline; the others keep everything.
sed -i "s/kept_pct: .*/kept_pct: ${KEPT_PCT}/" "${TEMP_DIR}/${CONFIG_PREFIX}_sp.yaml"
for cfg in nopruning infobatch dp; do
  sed -i "s/kept_pct: .*/kept_pct: 1.0/" "${TEMP_DIR}/${CONFIG_PREFIX}_${cfg}.yaml"
done

LOG_DIR="${OPERA_WORK_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/reproduce_fiqa_${MODEL_NAME}_${RUN_ID}.log"

# Experiment directory names, one per condition.
PRETRAIN_SAVE="fiqa_${MODEL_NAME}_pretrain_${RUN_ID}"
NOPRUNING_SAVE="fiqa_${MODEL_NAME}_nopruning_${RUN_ID}"
SP_SAVE="fiqa_${MODEL_NAME}_sp_${RUN_ID}"
INFOBATCH_SAVE="fiqa_${MODEL_NAME}_infobatch_${RUN_ID}"
DP_SAVE="fiqa_${MODEL_NAME}_dp_${RUN_ID}"

{
  echo "================================================"
  echo "FiQA reproduction for BAAI/bge-large-en-v1.5"
  echo "Train / Val / Test : $TRAIN_DATA / $VAL_DATA / $TEST_DATA"
  echo "Mining             : $MINING_MODE, range $MINING_RANGE"
  echo "Static kept_pct    : $KEPT_PCT"
  echo "Max steps          : $MAX_STEPS"
  echo "Batch size         : $BATCH_SIZE"
  echo "GPUs               : $NUM_GPUS"
  echo "================================================"
} | tee "$LOG"

# run_full <config_basename> <save_name>
# Runs the full finetuning pipeline for one condition.
run_full () {
  local cfg_name=$1
  local save_name=$2
  local cfg="${TEMP_DIR}/${cfg_name}.yaml"

  echo "[$(date)] [$save_name] prepare_ir" | tee -a "$LOG"
  python3 ./launcher.py --basecfg "$cfg" --save_name "$save_name" $DATA_PARAMS --task prepare_ir 2>&1 | tee -a "$LOG"

  echo "[$(date)] [$save_name] prune" | tee -a "$LOG"
  python3 ./launcher.py --basecfg "$cfg" --save_name "$save_name" --task prune 2>&1 | tee -a "$LOG"

  echo "[$(date)] [$save_name] mine" | tee -a "$LOG"
  python3 ./launcher.py --basecfg "$cfg" --save_name "$save_name" --task mine 2>&1 | tee -a "$LOG"

  echo "[$(date)] [$save_name] finetune ($NUM_GPUS GPUs, $MAX_STEPS steps)" | tee -a "$LOG"
  torchrun --nproc_per_node="$NUM_GPUS" ./launcher.py \
    --basecfg "$cfg" --save_name "$save_name" --task finetune --max_steps "$MAX_STEPS" 2>&1 | tee -a "$LOG"

  echo "[$(date)] [$save_name] evaluate" | tee -a "$LOG"
  python3 ./launcher.py --basecfg "$cfg" --save_name "$save_name" --task evaluate --max_steps "$MAX_STEPS" 2>&1 | tee -a "$LOG"
}

# Condition 1: pretrain (evaluate the base model, no finetuning).
echo "[$(date)] [$PRETRAIN_SAVE] prepare_ir" | tee -a "$LOG"
python3 ./launcher.py --basecfg "${TEMP_DIR}/${CONFIG_PREFIX}_pretrain.yaml" \
  --save_name "$PRETRAIN_SAVE" $DATA_PARAMS --task prepare_ir 2>&1 | tee -a "$LOG"

echo "[$(date)] [$PRETRAIN_SAVE] evaluate (base model)" | tee -a "$LOG"
python3 ./launcher.py --basecfg "${TEMP_DIR}/${CONFIG_PREFIX}_pretrain.yaml" \
  --save_name "$PRETRAIN_SAVE" --task evaluate 2>&1 | tee -a "$LOG"

# Conditions 2 to 5: full finetuning pipelines.
run_full "${CONFIG_PREFIX}_nopruning" "$NOPRUNING_SAVE"
run_full "${CONFIG_PREFIX}_sp"        "$SP_SAVE"
run_full "${CONFIG_PREFIX}_infobatch" "$INFOBATCH_SAVE"
run_full "${CONFIG_PREFIX}_dp"        "$DP_SAVE"

# Summarize Recall@20 and NDCG@10 for every condition.
echo "" | tee -a "$LOG"
echo "[$(date)] Summary" | tee -a "$LOG"
OPERA_WORK_ROOT="$OPERA_WORK_ROOT" \
PRETRAIN_SAVE="$PRETRAIN_SAVE" NOPRUNING_SAVE="$NOPRUNING_SAVE" SP_SAVE="$SP_SAVE" \
INFOBATCH_SAVE="$INFOBATCH_SAVE" DP_SAVE="$DP_SAVE" \
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import glob
import json
import os

work_root = os.environ["OPERA_WORK_ROOT"]
conditions = [
    ("pretrain",   os.environ["PRETRAIN_SAVE"]),
    ("no pruning", os.environ["NOPRUNING_SAVE"]),
    ("sp",         os.environ["SP_SAVE"]),
    ("infobatch",  os.environ["INFOBATCH_SAVE"]),
    ("dp (OPERA)", os.environ["DP_SAVE"]),
]

def read_metrics(save_name):
    exp_dir = os.path.join(work_root, save_name)
    # evaluate writes results.json (finetuned models) or
    # <model>-results.json (pretrain) somewhere under the experiment dir.
    files = sorted(glob.glob(os.path.join(exp_dir, "**", "*results.json"), recursive=True))
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)

header = f"{'Condition':<14}{'Recall@20':>12}{'NDCG@10':>12}"
print(header)
print("-" * len(header))
for label, save_name in conditions:
    metrics = read_metrics(save_name)
    if metrics is None:
        print(f"{label:<14}{'missing':>12}{'missing':>12}")
        continue
    r20 = metrics.get("Recall@20")
    n10 = metrics.get("NDCG@10")
    r20 = f"{r20:.4f}" if isinstance(r20, (int, float)) else "n/a"
    n10 = f"{n10:.4f}" if isinstance(n10, (int, float)) else "n/a"
    print(f"{label:<14}{r20:>12}{n10:>12}")
PY

echo "" | tee -a "$LOG"
echo "[$(date)] Done. Full log: $LOG" | tee -a "$LOG"
