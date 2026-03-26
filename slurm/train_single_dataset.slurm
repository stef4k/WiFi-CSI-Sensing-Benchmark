#!/bin/bash
#SBATCH --job-name=sensefi-dataset
#SBATCH --output=slurm_logs/out/%x.%j.out
#SBATCH --error=slurm_logs/err/%x.%j.err
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

if [ -z "${MODEL:-}" ]; then
  echo "MODEL env var is required. Example: sbatch --export=ALL,MODEL=GRU,DATASET=Widar slurm/train_single_dataset.slurm"
  exit 1
fi

if [ -z "${DATASET:-}" ]; then
  echo "DATASET env var is required. Example: sbatch --export=ALL,MODEL=GRU,DATASET=Widar slurm/train_single_dataset.slurm"
  exit 1
fi

ROOT_DIR="${ROOT_DIR:-$PWD}"
RESULTS_DIR="${RESULTS_DIR:-results}"

module purge
module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sensefi39

cd "$ROOT_DIR"

echo "Running MODEL=$MODEL DATASET=$DATASET"
echo "Results directory: $RESULTS_DIR"

python -u run.py --model "$MODEL" --dataset "$DATASET" --results-dir "$RESULTS_DIR"
