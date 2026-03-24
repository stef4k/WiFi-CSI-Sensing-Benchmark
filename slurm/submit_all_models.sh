#!/bin/bash
set -euo pipefail

mkdir -p slurm_logs/out slurm_logs/err

ROOT_DIR="${ROOT_DIR:-$PWD}"
RESULTS_DIR="${RESULTS_DIR:-results}"

MODELS=(
  "MLP"
  "LeNet"
  "ResNet18"
  "ResNet50"
  "ResNet101"
  "RNN"
  "GRU"
  "LSTM"
  "BiLSTM"
  "CNN+GRU"
  "ViT"
)

for model in "${MODELS[@]}"; do
  safe_name="${model//+/plus}"
  sbatch \
    --job-name="sensefi-${safe_name}" \
    --export=ALL,MODEL="$model",ROOT_DIR="$ROOT_DIR",RESULTS_DIR="$RESULTS_DIR" \
    slurm/train_model_all_datasets.slurm
done