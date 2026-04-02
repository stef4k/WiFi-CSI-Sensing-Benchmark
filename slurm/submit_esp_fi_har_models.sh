#!/bin/bash
# Submit one SLURM job per model for ESP-Fi-HAR training.
# Each job runs LOSO (8 folds) + leave-one-env (4 folds) + random-split (5 seeds).
#
# Partition assignment:
#   gpua100 — ResNet18, ResNet50, ResNet101, ViT  (heavy conv / attention models)
#   gpu     — MLP, LeNet, RNN, GRU, LSTM, BiLSTM, CNN+GRU  (lighter models)
#
# Usage:
#   bash slurm/submit_esp_fi_har_models.sh
#   ROOT_DIR=/custom/path RESULTS_DIR=my_results bash slurm/submit_esp_fi_har_models.sh

set -euo pipefail

mkdir -p slurm_logs/out slurm_logs/err

ROOT_DIR="${ROOT_DIR:-$PWD}"
RESULTS_DIR="${RESULTS_DIR:-results}"

# Models that need A100s (large conv / attention — most compute-intensive)
MODELS_A100=(
  "ResNet18"
  "ResNet50"
  "ResNet101"
  "ViT"
)

# Lighter models that run fine on the regular GPU partition
MODELS_GPU=(
  "MLP"
  "LeNet"
  "RNN"
  "GRU"
  "LSTM"
  "BiLSTM"
  "CNN+GRU"
)

submit() {
  local model="$1"
  local partition="$2"
  local safe_name="${model//+/plus}"
  local job_id
  job_id=$(sbatch \
    --job-name="espfi-${safe_name}" \
    --partition="$partition" \
    --export=ALL,MODEL="$model",ROOT_DIR="$ROOT_DIR",RESULTS_DIR="$RESULTS_DIR" \
    slurm/train_esp_fi_har.slurm | awk '{print $NF}')
  echo "Submitted MODEL=$model  partition=$partition  job_id=$job_id"
}

for model in "${MODELS_A100[@]}"; do
  submit "$model" "gpua100"
done

for model in "${MODELS_GPU[@]}"; do
  submit "$model" "gpu"
done

echo "All ESP-Fi-HAR jobs submitted."
