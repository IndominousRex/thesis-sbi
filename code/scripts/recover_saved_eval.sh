#!/bin/bash -l
# ==============================================================================
# SLURM wrapper for recovering eval metrics/plots from saved FNPE/Simformer runs.
#
# Run from inside the experiments directory or pass the experiments dir explicitly:
#   sbatch ../scripts/recover_saved_eval.sh
#   sbatch ../scripts/recover_saved_eval.sh /bigwork/.../code/experiments bench_budget_v2_mu
# Parallel arrays:
#   sbatch --array=0-3 ../scripts/recover_saved_eval.sh /bigwork/.../code/experiments bench_budget_v2_mu_s42 cuda
#   sbatch --array=0-7%4 ../scripts/recover_saved_eval.sh /bigwork/.../code/experiments bench_budget_v2_mu_s42 cuda
# ==============================================================================

#SBATCH --job-name=recover_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=recover_eval_%j.out
#SBATCH --error=recover_eval_%j.err

set -euo pipefail

EXPERIMENTS_DIR=${1:-"$PWD"}
MATCH_PREFIX=${2:-"bench_budget_v2_mu_s42"}
DEVICE=${3:-"cuda"}

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/recover_saved_eval.py" ]; then
    CODE_DIR="${SLURM_SUBMIT_DIR}"
elif [ -d "/bigwork/nhkbarit/thesis-code/code" ]; then
    CODE_DIR="/bigwork/nhkbarit/thesis-code/code"
else
    echo "Could not determine code directory."
    echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
    exit 1
fi

echo "=================================================="
echo "Recover Saved Eval"
echo "=================================================="
echo "Date:            $(date)"
echo "Node:            $(hostname)"
echo "Job ID:          ${SLURM_JOB_ID:-local}"
echo "Array Task:      ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"
echo "Experiments Dir: ${EXPERIMENTS_DIR}"
echo "Match Prefix:    ${MATCH_PREFIX}"
echo "Device:          ${DEVICE}"
echo "=================================================="

cd "${CODE_DIR}"

module load Miniforge3
ENV_PREFIX="${SBI_ENV_PREFIX:-/software/NHKB22930/nhkbarit/conda_envs/npe}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

unset PYTHONHOME || true
unset PYTHONPATH || true
export PATH="${ENV_PREFIX}/bin:${PATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}

conda run -p "${ENV_PREFIX}" --no-capture-output python scripts/recover_saved_eval.py \
    --experiments-dir "${EXPERIMENTS_DIR}" \
    --match-prefix "${MATCH_PREFIX}" \
    --device "${DEVICE}" \
    --shard-index "${SHARD_INDEX}" \
    --num-shards "${NUM_SHARDS}"

echo "=================================================="
echo "[$(date)] Recovery completed"
echo "=================================================="
