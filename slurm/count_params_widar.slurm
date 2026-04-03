#!/bin/bash
#SBATCH --job-name=count-params-widar
#SBATCH --output=slurm_logs/out/%x.%j.out
#SBATCH --error=slurm_logs/err/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PWD}"
RESULTS_DIR="${RESULTS_DIR:-results}"

module purge
module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sensefi39

cd "$ROOT_DIR"

mkdir -p slurm_logs/out slurm_logs/err

echo "Counting parameters for all Widar models"
echo "Results dir: $RESULTS_DIR"
echo "Node: $(hostname)"

python count_params_widar.py \
    --root ./Data/ \
    --results-dir "$RESULTS_DIR"

echo "Done."
