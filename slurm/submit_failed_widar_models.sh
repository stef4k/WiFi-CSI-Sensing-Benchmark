#!/bin/bash
set -euo pipefail

mkdir -p slurm_logs/out slurm_logs/err

ROOT_DIR="${ROOT_DIR:-$PWD}"
RESULTS_DIR="${RESULTS_DIR:-results}"
PARTITION="${PARTITION:-gpua100}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"

FAILED_WIDAR_MODELS=(
  "BiLSTM"
  "CNN+GRU"
  "GRU"
  "LSTM"
  "LeNet"
  "RNN"
  "ResNet101"
  "ResNet18"
  "ResNet50"
  "ViT"
)

echo "Submitting Widar-only jobs for models that did not finish in the previous sweep."
echo "Partition: $PARTITION"
echo "Time limit: $TIME_LIMIT"
echo "Results directory: $RESULTS_DIR"

for model in "${FAILED_WIDAR_MODELS[@]}"; do
  safe_name="${model//+/plus}"
  sbatch \
    --job-name="sensefi-widar-${safe_name}" \
    --partition="$PARTITION" \
    --gres="gpu:${GPUS_PER_JOB}" \
    --time="$TIME_LIMIT" \
    --export=ALL,MODEL="$model",DATASET="Widar",ROOT_DIR="$ROOT_DIR",RESULTS_DIR="$RESULTS_DIR" \
    slurm/train_single_dataset.slurm
done
