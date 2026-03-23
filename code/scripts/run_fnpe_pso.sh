#!/bin/bash -l
# ==============================================================================
# SLURM script for running FNPE with PSO-optimized vehicle parameters
#
# Usage:
#   sbatch run_fnpe_pso.sh                                           # defaults
#   sbatch run_fnpe_pso.sh fnpe_pso_v1 10000 1000 20000 1500         # name/budget/Tseg/numsim/pilot_length
#   sbatch run_fnpe_pso.sh fnpe_norm_off 10000 1000 20000 1500 yes   # 6th arg = skip normalization
# ==============================================================================

#SBATCH --job-name=fnpe_pso
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=fnpe_pso_%j.out
#SBATCH --error=fnpe_pso_%j.err

# ==============================================================================
# Configurable arguments (with defaults)
# ==============================================================================
EXP_NAME=${1:-fnpe_pso_optimized}
FNPE_NUM_SIM=${2:-10000}
T_SEG=${3:-1000}
NUM_SIM=${4:-20000}
PILOT_LENGTH=${5:-500}
SKIP_NORM=${6:-no}   # "yes" to disable normalization

# Path to PSO results (relative to code dir)
PSO_JSON="notebooks/experiments/pso_global_log_optimization_results.json"

# ==============================================================================
# Environment setup
# ==============================================================================
set -e

echo "=================================================="
echo "FNPE with PSO-Optimized Parameters"
echo "=================================================="
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Exp Name:     ${EXP_NAME}"
echo "FNPE Budget:  ${FNPE_NUM_SIM}"
echo "Num Sims:     ${NUM_SIM}"
echo "T_seg:        ${T_SEG}"
echo "Pilot Length: ${PILOT_LENGTH}"
echo "Skip Norm:    ${SKIP_NORM}"
echo "PSO JSON:     ${PSO_JSON}"
echo "=================================================="

# --- Go to code directory ---
cd /bigwork/nhkbarit/thesis-code/code

# --- Load environment ---
module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

# --- JAX configuration ---
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORM_NAME || true

# ==============================================================================
# Build optional flags
# ==============================================================================
EXTRA_FLAGS="--fnpe-clip-samples"

if [ "${SKIP_NORM}" = "yes" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --fnpe-skip-normalize"
    echo "[INFO] Normalization DISABLED"
fi

# ==============================================================================
# Run FNPE with PSO params
# ==============================================================================
srun python run_fnpe_pso.py \
    --pso-json ${PSO_JSON} \
    --exp-name ${EXP_NAME} \
    --device cuda \
    --fnpe-num-sim ${FNPE_NUM_SIM} \
    --num-sim ${NUM_SIM} \
    --T-seg ${T_SEG} \
    --params mu \
    --num-epochs 200 \
    --batch-size 512 \
    --fnpe-steps-per-epoch 10000 \
    --fnpe-diffusion-steps 500 \
    --fnpe-t-min 0.05 \
    --fnpe-pilot-length ${PILOT_LENGTH} \
    --data-dir ../data/measurements \
    --K-ppc 300 \
    ${EXTRA_FLAGS}

echo "=================================================="
echo "[$(date)] Completed"
echo "=================================================="
