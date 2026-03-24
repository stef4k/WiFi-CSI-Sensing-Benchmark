#!/bin/bash
#SBATCH --job-name=sensefi-model
#SBATCH --output=slurm_logs/out/%x.%j.out
#SBATCH --error=slurm_logs/err/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

if [ -z "${MODEL:-}" ]; then
  echo "MODEL env var is required. Example: sbatch --export=ALL,MODEL=MLP slurm/train_model_all_datasets.slurm"
  exit 1
fi

ROOT_DIR="${ROOT_DIR:-$PWD}"
RESULTS_DIR="${RESULTS_DIR:-results}"
DATASETS=("UT_HAR_data" "NTU-Fi-HumanID" "NTU-Fi_HAR" "Widar")

module purge
module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sensefi39

cd "$ROOT_DIR"

echo "Running MODEL=$MODEL on datasets: ${DATASETS[*]}"
echo "Results directory: $RESULTS_DIR"

for dataset in "${DATASETS[@]}"; do
  echo "[START] model=$MODEL dataset=$dataset"
  python run.py --model "$MODEL" --dataset "$dataset" --results-dir "$RESULTS_DIR"
  echo "[DONE] model=$MODEL dataset=$dataset"
done

echo "Sweep done for MODEL=$MODEL"
