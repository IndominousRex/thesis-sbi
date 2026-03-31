#!/bin/bash -l
# ==============================================================================
# SLURM job-array script for comparison experiments.
#
# Submit this script with sbatch. It is designed to run 4 array tasks in
# parallel by default, one method per task, one GPU per task.
#
# By default each task regenerates its datasets deterministically from the
# configured seeds. Dataset caching/reuse is now opt-in at the Python entrypoint.
# ==============================================================================

#SBATCH --job-name=sim_cmp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=sim_cmp_%A_%a.out
#SBATCH --error=sim_cmp_%A_%a.err

set -euo pipefail

NUM_SIMULATIONS=${1:-2000}
T_SEG=${2:-1000}
EXP_NAME=${3:-simformer_compare}
MODE=${4:-no}
METHODS=${5:-"npe npse fnpe simformer"}
SEED=${6:-42}
PARAMS=${7:-"mu,cd,m"}
REQUESTED_BUDGET_STEPS=${8:-""}

MODE_FLAG=""
if [ "${MODE}" = "yes" ]; then
    MODE_FLAG="--quick"
elif [ "${MODE}" = "smoke" ]; then
    MODE_FLAG="--smoke"
elif [ "${MODE}" != "no" ]; then
    echo "Unsupported mode: ${MODE}"
    exit 1
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/run_simulation_comparison.py" ]; then
    CODE_DIR="${SLURM_SUBMIT_DIR}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/run_simulation_comparison.py" ]; then
    CODE_DIR=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)
elif [ -d "/bigwork/nhkbarit/thesis-code/code" ]; then
    CODE_DIR="/bigwork/nhkbarit/thesis-code/code"
else
    echo "Could not determine code directory."
    echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
    exit 1
fi
RUN_GROUP="${EXP_NAME}_A${SLURM_ARRAY_JOB_ID:-local}"

read -r -a METHOD_ARRAY <<< "${METHODS}"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_METHODS=${#METHOD_ARRAY[@]}

if [ "${NUM_METHODS}" -gt 4 ]; then
    echo "This script is fixed to 4 parallel array tasks. Got ${NUM_METHODS} methods."
    exit 1
fi

if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${NUM_METHODS}" ]; then
    echo "No method assigned for array task ${TASK_ID}; exiting."
    exit 0
fi

METHOD="${METHOD_ARRAY[${TASK_ID}]}"

echo "=================================================="
echo "Comparison Array Task"
echo "=================================================="
echo "Date:            $(date)"
echo "Node:            $(hostname)"
echo "Job ID:          ${SLURM_JOB_ID:-local}"
echo "Array Job ID:    ${SLURM_ARRAY_JOB_ID:-local}"
echo "Array Task ID:   ${TASK_ID}"
echo "Method:          ${METHOD}"
echo "Run Group:       ${RUN_GROUP}"
echo "Num Simulations: ${NUM_SIMULATIONS}"
echo "T_seg:           ${T_SEG}"
echo "Mode:            ${MODE}"
echo "Methods:         ${METHODS}"
echo "Num Methods:     ${NUM_METHODS}"
echo "Seed:            ${SEED}"
echo "Params:          ${PARAMS}"
if [ -n "${REQUESTED_BUDGET_STEPS}" ]; then
    echo "Budget Steps:    ${REQUESTED_BUDGET_STEPS}"
fi
echo "=================================================="

cd "${CODE_DIR}"

module load Miniforge3
ENV_PREFIX="${SBI_ENV_PREFIX:-/software/NHKB22930/nhkbarit/conda_envs/npe}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"

unset PYTHONHOME || true
unset PYTHONPATH || true
export PATH="${ENV_PREFIX}/bin:${PATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Python:          ${PYTHON_BIN}"
conda run -p "${ENV_PREFIX}" --no-capture-output python -c "import sys; print('sys.executable:  ', sys.executable); print('sys.prefix:      ', sys.prefix)"
conda run -p "${ENV_PREFIX}" --no-capture-output python -c "import torch; print('torch version:    ', torch.__version__)"
conda run -p "${ENV_PREFIX}" --no-capture-output python -c "import jax; print('jax backend:      ', jax.default_backend())"

PY_CMD=(
    conda run -p "${ENV_PREFIX}" --no-capture-output python
    scripts/run_simulation_comparison.py
    --exp-name "${RUN_GROUP}"
    --num-simulations "${NUM_SIMULATIONS}"
    --T-seg "${T_SEG}"
    --device cuda
    --methods "${METHOD}"
    --seed "${SEED}"
    --params "${PARAMS}"
    --no-summary
)

if [ -n "${REQUESTED_BUDGET_STEPS}" ]; then
    PY_CMD+=(--requested-budget-steps "${REQUESTED_BUDGET_STEPS}")
fi

if [ -n "${MODE_FLAG}" ]; then
    PY_CMD+=("${MODE_FLAG}")
fi

srun "${PY_CMD[@]}"

echo "=================================================="
echo "[$(date)] Task completed"
echo "=================================================="
