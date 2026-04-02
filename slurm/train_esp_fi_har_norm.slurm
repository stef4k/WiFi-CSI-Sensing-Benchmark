#!/bin/bash
#SBATCH --job-name=espfi-norm
#SBATCH --output=slurm_logs/out/%x.%j.out
#SBATCH --error=slurm_logs/err/%x.%j.err
#SBATCH --gres=gpu:1
# Partition is set by the submit script (--partition flag on sbatch).
#SBATCH --exclude=ruche-gpu13
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

if [ -z "${MODEL:-}" ]; then
  echo "MODEL env var is required."
  exit 1
fi
if [ -z "${NORM:-}" ]; then
  echo "NORM env var is required."
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

echo "ESP-Fi-HAR norm experiment — MODEL=$MODEL  NORM=$NORM"
echo "Results directory: $RESULTS_DIR"

run_fold() {
  local strategy="$1"
  local held_out="$2"
  echo "[START] strategy=$strategy held_out=$held_out"
  python -u run_esp_fi_har.py \
    --model    "$MODEL" \
    --strategy "$strategy" \
    --held-out "$held_out" \
    --norm     "$NORM" \
    --results-dir "$RESULTS_DIR"
  echo "[DONE]  strategy=$strategy held_out=$held_out"
}

# ------------------------------------------------------------------
# Strategy 1: Leave One Subject Out (LOSO) — 8 folds
# ------------------------------------------------------------------
echo "=== Strategy: loso ==="
for subject in 1 2 3 4 5 6 7 8; do
  run_fold loso "$subject"
done

# ------------------------------------------------------------------
# Strategy 2: Leave One Environment Out — 4 folds
# ------------------------------------------------------------------
echo "=== Strategy: leave-one-env ==="
for env in 1 2 3 4; do
  run_fold leave-one-env "$env"
done

# ------------------------------------------------------------------
# Strategy 3: Random stratified split — 5 seeds
# ------------------------------------------------------------------
echo "=== Strategy: random-split ==="
for seed in 1 2 3 4 5; do
  run_fold random-split "$seed"
done

echo "All ESP-Fi-HAR norm folds done for MODEL=$MODEL NORM=$NORM"
