#!/bin/bash -l
# ==============================================================================
# SLURM batch script to prepare the shared cached dataset for comparison runs.
# ==============================================================================

#SBATCH --job-name=sim_cmp_prep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=sim_cmp_prep_%j.out
#SBATCH --error=sim_cmp_prep_%j.err

set -euo pipefail

NUM_SIMULATIONS=${1:-2000}
T_SEG=${2:-1000}
EXP_NAME=${3:-simformer_compare}
MODE=${4:-no}
SEED=${5:-42}

MODE_FLAG=""
if [ "${MODE}" = "yes" ]; then
    MODE_FLAG="--quick"
elif [ "${MODE}" = "smoke" ]; then
    MODE_FLAG="--smoke"
elif [ "${MODE}" != "no" ]; then
    echo "Unsupported mode: ${MODE}"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

echo "=================================================="
echo "Prepare Shared Dataset"
echo "=================================================="
echo "Date:            $(date)"
echo "Node:            $(hostname)"
echo "Job ID:          ${SLURM_JOB_ID:-local}"
echo "Num Simulations: ${NUM_SIMULATIONS}"
echo "T_seg:           ${T_SEG}"
echo "Exp Name:        ${EXP_NAME}"
echo "Mode:            ${MODE}"
echo "Seed:            ${SEED}"
echo "=================================================="

cd "${CODE_DIR}"

module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

export XLA_PYTHON_CLIENT_PREALLOCATE=false

srun python scripts/prepare_shared_dataset.py \
    --exp-name "${EXP_NAME}" \
    --num-simulations "${NUM_SIMULATIONS}" \
    --T-seg "${T_SEG}" \
    --device cuda \
    --seed "${SEED}" \
    ${MODE_FLAG}

echo "=================================================="
echo "[$(date)] Shared dataset prepared"
echo "=================================================="
