#!/bin/bash
#
# Run the full OPERA pipeline for a single config and dataset:
#   prepare_ir -> prune -> mine -> finetune -> evaluate -> gather_results
#
# Usage:
#   ./run.sh <config> <save_name> <train_data> <val_data> <test_data>
#
# Example:
#   ./run.sh configs/opera_qwen_0.6B_dp.yaml nfcorpus_dp \
#            beir/nfcorpus/train beir/nfcorpus/dev beir/nfcorpus/test
#
# When a dataset has no test qrels, pass the dev split as the test split.
#
# Environment variables:
#   OPERA_RAW_DATA_DIR  directory for raw and preprocessed data (required)
#   OPERA_WORK_ROOT     directory for checkpoints and results (required)
#   NUM_GPUS            number of GPUs for distributed finetuning (default: 8)
#   MAX_STEPS           training steps; overrides the value in the config (default: 4000)

set -e

if [ $# -lt 5 ]; then
  echo "Usage: $0 <config> <save_name> <train_data> <val_data> <test_data>"
  echo "Example: $0 configs/opera_qwen_0.6B_dp.yaml nfcorpus_dp beir/nfcorpus/train beir/nfcorpus/dev beir/nfcorpus/test"
  exit 1
fi

CONFIG=$1
SAVE_NAME=$2
TRAIN_DATA=$3
VAL_DATA=$4
TEST_DATA=$5

: "${OPERA_RAW_DATA_DIR:?Set OPERA_RAW_DATA_DIR to the directory for raw and preprocessed data}"
: "${OPERA_WORK_ROOT:?Set OPERA_WORK_ROOT to the directory for checkpoints and results}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_STEPS="${MAX_STEPS:-4000}"

DATA_PARAMS="--train_data $TRAIN_DATA --val_data $VAL_DATA --test_data $TEST_DATA"

echo "[$(date)] Preparing data..."
python3 launcher.py -c "$CONFIG" -s "$SAVE_NAME" $DATA_PARAMS -t prepare_ir

echo "[$(date)] Offline pruning..."
python3 launcher.py -c "$CONFIG" -s "$SAVE_NAME" -t prune

echo "[$(date)] Mining negatives..."
python3 launcher.py -c "$CONFIG" -s "$SAVE_NAME" -t mine

echo "[$(date)] Finetuning on $NUM_GPUS GPU(s) for $MAX_STEPS steps..."
torchrun --nproc_per_node="$NUM_GPUS" launcher.py \
  -c "$CONFIG" -s "$SAVE_NAME" -t finetune --max_steps "$MAX_STEPS"

echo "[$(date)] Evaluating..."
python3 launcher.py -c "$CONFIG" -s "$SAVE_NAME" -t evaluate --max_steps "$MAX_STEPS"

echo "[$(date)] Gathering results..."
python3 launcher.py -c "$CONFIG" -s "$SAVE_NAME" -t gather_results

echo "[$(date)] Done. Results are under \$OPERA_WORK_ROOT/$SAVE_NAME"
