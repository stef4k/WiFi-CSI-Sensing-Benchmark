#!/bin/bash
# Submit one SLURM job per (model, norm) combination for the ESP-Fi-HAR
# normalization comparison experiment.
#
# Each job runs all 17 folds: LOSO (8) + leave-one-env (4) + random-split (5).
# Results land in results/esp_fi_har_norm_comparison.csv.
#
# Partition assignment
# --------------------
#   gpua100 — ResNet18, ResNet50, ResNet101, ViT  (compute-heavy)
#   gpu     — MLP, LeNet, RNN, GRU, LSTM, BiLSTM, CNN+GRU
#
# Jobs are interleaved across the two partitions to spread the load.
#
# Usage:
#   bash slurm/submit_esp_fi_har_norm_jobs.sh
#   RESULTS_DIR=my_results bash slurm/submit_esp_fi_har_norm_jobs.sh

set -euo pipefail

mkdir -p slurm_logs/out slurm_logs/err

ROOT_DIR="${ROOT_DIR:-$PWD}"
RESULTS_DIR="${RESULTS_DIR:-results}"

NORMS=(
  "per-sample"
  "global"
  "global-zscore"
  "per-subcarrier"
  "per-subcarrier-zscore"
  "per-subject"
  "per-environment"
  "robust"
)

# Models that benefit from A100 GPUs
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
  local norm="$2"
  local partition="$3"
  local safe_model="${model//+/plus}"
  local safe_norm="${norm//-/_}"
  local job_id
  job_id=$(sbatch \
    --job-name="espfi-norm-${safe_model}-${safe_norm}" \
    --partition="$partition" \
    --export=ALL,MODEL="$model",NORM="$norm",ROOT_DIR="$ROOT_DIR",RESULTS_DIR="$RESULTS_DIR" \
    slurm/train_esp_fi_har_norm.slurm | awk '{print $NF}')
  echo "Submitted  MODEL=$model  NORM=$norm  partition=$partition  job_id=$job_id"
}

total=0
for model in "${MODELS_A100[@]}"; do
  for norm in "${NORMS[@]}"; do
    submit "$model" "$norm" "gpua100"
    total=$((total + 1))
  done
done

for model in "${MODELS_GPU[@]}"; do
  for norm in "${NORMS[@]}"; do
    submit "$model" "$norm" "gpu"
    total=$((total + 1))
  done
done

echo ""
echo "Submitted $total jobs total  (${#MODELS_A100[@]} models × gpua100  +  ${#MODELS_GPU[@]} models × gpu  ×  ${#NORMS[@]} norms)"
echo "Results will be written to: $RESULTS_DIR/esp_fi_har_norm_comparison.csv"
