#!/bin/bash -l
# ==============================================================================
# Slurm array runner for benchmark manifests.
#
# Submit via submit_simulation_benchmark_array.py. Each array task executes one
# method/cell from a manifest and Slurm handles backfilling as tasks complete.
# ==============================================================================

#SBATCH --job-name=simbench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-0%1
#SBATCH --output=simbench_%A_%a.out
#SBATCH --error=simbench_%A_%a.err

set -euo pipefail

MANIFEST_PATH=${1:?manifest path required}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

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

echo "=================================================="
echo "Benchmark Manifest Array Task"
echo "=================================================="
echo "Date:          $(date)"
echo "Node:          $(hostname)"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Array Task ID: ${TASK_ID}"
echo "Manifest:      ${MANIFEST_PATH}"
echo "Code dir:      ${CODE_DIR}"
echo "GPU request:   ${SLURM_JOB_GRES:-gpu:1}"
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
conda run -p "${ENV_PREFIX}" --no-capture-output python -c "import torch; print('torch version:    ', torch.__version__)"
conda run -p "${ENV_PREFIX}" --no-capture-output python -c "import jax; print('jax backend:      ', jax.default_backend())"

srun conda run -p "${ENV_PREFIX}" --no-capture-output python \
    scripts/run_simulation_benchmark_manifest_cell.py \
    --manifest "${MANIFEST_PATH}" \
    --index "${TASK_ID}"

echo "=================================================="
echo "[$(date)] Task completed"
echo "=================================================="
