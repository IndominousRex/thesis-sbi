#!/bin/bash -l
# ==============================================================================
# SLURM script for running a single comparison-style experiment.
#
# This mirrors the comparison array worker, but runs exactly one method on one
# GPU with no array/parallelization.
#
# Usage:
#   sbatch scripts/run_simulation_single.sh
#   sbatch scripts/run_simulation_single.sh 20000 1000 cmp_single no npe 42
#   sbatch scripts/run_simulation_single.sh 20000 1000 cmp_single no simformer 42 "mu"
#
# For a guaranteed >=20 GB GPU based on the cluster guide, override on submit:
#   sbatch --gres=gpu:a100m40:1 scripts/run_simulation_single.sh ...
# ==============================================================================

#SBATCH --job-name=sim_cmp_1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=sim_cmp_1_%j.out
#SBATCH --error=sim_cmp_1_%j.err

set -euo pipefail

NUM_SIMULATIONS=${1:-2000}
T_SEG=${2:-1000}
EXP_NAME=${3:-simformer_compare_single}
MODE=${4:-no}
METHOD=${5:-npe}
SEED=${6:-42}
PARAMS=${7:-"mu,cd,m"}

MODE_FLAG=""
if [ "${MODE}" = "yes" ]; then
    MODE_FLAG="--quick"
elif [ "${MODE}" = "smoke" ]; then
    MODE_FLAG="--smoke"
elif [ "${MODE}" != "no" ]; then
    echo "Unsupported mode: ${MODE}"
    exit 1
fi

case "${METHOD}" in
    npe|npse|fnpe|simformer)
        ;;
    *)
        echo "Unsupported method: ${METHOD}"
        exit 1
        ;;
esac

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

RUN_NAME="${EXP_NAME}_J${SLURM_JOB_ID:-local}"

echo "=================================================="
echo "Single Comparison Task"
echo "=================================================="
echo "Date:            $(date)"
echo "Node:            $(hostname)"
echo "Job ID:          ${SLURM_JOB_ID:-local}"
echo "Method:          ${METHOD}"
echo "Run Name:        ${RUN_NAME}"
echo "Num Simulations: ${NUM_SIMULATIONS}"
echo "T_seg:           ${T_SEG}"
echo "Mode:            ${MODE}"
echo "Seed:            ${SEED}"
echo "Params:          ${PARAMS}"
echo "=================================================="

cd "${CODE_DIR}"

module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

export XLA_PYTHON_CLIENT_PREALLOCATE=false

srun python scripts/run_simulation_comparison.py \
    --exp-name "${RUN_NAME}" \
    --num-simulations "${NUM_SIMULATIONS}" \
    --T-seg "${T_SEG}" \
    --device cuda \
    --methods "${METHOD}" \
    --seed "${SEED}" \
    --params "${PARAMS}" \
    --no-summary \
    --independent-datasets \
    ${MODE_FLAG}

echo "=================================================="
echo "[$(date)] Task completed"
echo "=================================================="
