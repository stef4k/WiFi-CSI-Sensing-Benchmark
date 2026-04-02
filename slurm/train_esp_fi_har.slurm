#!/bin/bash
#SBATCH --job-name=espfi-model
#SBATCH --output=slurm_logs/out/%x.%j.out
#SBATCH --error=slurm_logs/err/%x.%j.err
#SBATCH --gres=gpu:1
# Partition is set by the submit script (--partition flag on sbatch).
# Exclude ruche-gpu13: its CUDA driver is incompatible with the sensefi39 env.
#SBATCH --exclude=ruche-gpu13
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

if [ -z "${MODEL:-}" ]; then
  echo "MODEL env var is required."
  echo "Example: sbatch --export=ALL,MODEL=GRU slurm/train_esp_fi_har.slurm"
  exit 1
fi

ROOT_DIR="${ROOT_DIR:-$PWD}"
RESULTS_DIR="${RESULTS_DIR:-results}"

module purge
module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sensefi39

cd "$ROOT_DIR"

mkdir -p slurm_logs/out slurm_logs/err

echo "ESP-Fi-HAR training — MODEL=$MODEL"
echo "Results directory: $RESULTS_DIR"

# ------------------------------------------------------------------
# Strategy 1: Leave One Subject Out (LOSO) — 8 folds (subjects 1-8)
# ------------------------------------------------------------------
echo "=== Strategy: loso ==="
for subject in 1 2 3 4 5 6 7 8; do
  echo "[START] loso subject=$subject"
  python -u run_esp_fi_har.py \
    --model "$MODEL" \
    --strategy loso \
    --held-out "$subject" \
    --results-dir "$RESULTS_DIR"
  echo "[DONE]  loso subject=$subject"
done

# ------------------------------------------------------------------
# Strategy 2: Leave One Environment Out — 4 folds (envs 1-4)
# ------------------------------------------------------------------
echo "=== Strategy: leave-one-env ==="
for env in 1 2 3 4; do
  echo "[START] leave-one-env env=$env"
  python -u run_esp_fi_har.py \
    --model "$MODEL" \
    --strategy leave-one-env \
    --held-out "$env" \
    --results-dir "$RESULTS_DIR"
  echo "[DONE]  leave-one-env env=$env"
done

# ------------------------------------------------------------------
# Strategy 3: Random stratified split — 5 seeds for stable estimate
# ------------------------------------------------------------------
echo "=== Strategy: random-split ==="
for seed in 1 2 3 4 5; do
  echo "[START] random-split seed=$seed"
  python -u run_esp_fi_har.py \
    --model "$MODEL" \
    --strategy random-split \
    --held-out "$seed" \
    --results-dir "$RESULTS_DIR"
  echo "[DONE]  random-split seed=$seed"
done

echo "All ESP-Fi-HAR folds done for MODEL=$MODEL"
