#!/bin/bash -l
# ==============================================================================
# Slurm array runner for rerunning incomplete experiments from saved configs.
#
# The input is a plain text file containing one config JSON path per line. Each
# array task loads the Nth config path, then launches a fresh train+eval rerun
# with the same saved hyperparameters/seeds but no checkpoint reuse and no
# dataset caching.
#
# Example:
#   sbatch --array=0-49%50 \
#     --output=/bigwork/.../code/experiments/slurm/rerun_cfg_%A_%a.out \
#     --error=/bigwork/.../code/experiments/slurm/rerun_cfg_%A_%a.err \
#     scripts/run_rerun_from_config_array.sh /bigwork/.../code/rerun_batch1.txt
# ==============================================================================

#SBATCH --job-name=rerun_cfg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-0%1
#SBATCH --output=rerun_cfg_%A_%a.out
#SBATCH --error=rerun_cfg_%A_%a.err

set -euo pipefail

CONFIG_LIST=${1:?config-list path required}
RESULTS_ROOT_ARG=${2:-""}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/rerun_experiment_from_config.py" ]; then
    CODE_DIR="${SLURM_SUBMIT_DIR}"
elif [ -d "/bigwork/nhkbarit/thesis-code/code" ]; then
    CODE_DIR="/bigwork/nhkbarit/thesis-code/code"
else
    echo "Could not determine code directory."
    echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
    exit 1
fi

if [ -n "${RESULTS_ROOT_ARG}" ]; then
    RESULTS_ROOT="${RESULTS_ROOT_ARG}"
else
    RESULTS_ROOT="${CODE_DIR}/experiments"
fi

if [ ! -f "${CONFIG_LIST}" ]; then
    echo "Config list not found: ${CONFIG_LIST}"
    exit 1
fi

CFG_PATH=$(sed -n "$((TASK_ID + 1))p" "${CONFIG_LIST}")
if [ -z "${CFG_PATH}" ]; then
    echo "No config assigned for array task ${TASK_ID}; exiting."
    exit 0
fi
if [ ! -f "${CFG_PATH}" ]; then
    echo "Config path does not exist: ${CFG_PATH}"
    exit 1
fi

echo "=================================================="
echo "Rerun-From-Config Array Task"
echo "=================================================="
echo "Date:          $(date)"
echo "Node:          $(hostname)"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Array Task ID: ${TASK_ID}"
echo "Config list:   ${CONFIG_LIST}"
echo "Config path:   ${CFG_PATH}"
echo "Code dir:      ${CODE_DIR}"
echo "Results root:  ${RESULTS_ROOT}"
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

echo "Python:        ${ENV_PREFIX}/bin/python"
conda run -p "${ENV_PREFIX}" --no-capture-output python -c "import sys; print('sys.executable:  ', sys.executable); print('sys.prefix:      ', sys.prefix)"
conda run -p "${ENV_PREFIX}" --no-capture-output python -c "import torch; print('torch version:    ', torch.__version__); print('cuda available:   ', torch.cuda.is_available())"
conda run -p "${ENV_PREFIX}" --no-capture-output python -c "import jax; print('jax backend:      ', jax.default_backend())"

srun conda run -p "${ENV_PREFIX}" --no-capture-output python \
    "${CODE_DIR}/scripts/rerun_experiment_from_config.py" \
    --config "${CFG_PATH}" \
    --results-root "${RESULTS_ROOT}" \
    --device cuda

echo "=================================================="
echo "[$(date)] Task completed"
echo "=================================================="
