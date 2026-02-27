#!/bin/bash -l
# ==============================================================================
# SLURM script for Advanced PSO optimisation with global tire params
#
# Optimises all vehicle/tire parameters jointly across 7 trajectories.
# Only mu (friction) varies per trajectory. Total: 23 parameters.
#
# Features: lbest→gbest topology switch, inertia schedule, LHS init,
#           stagnation reinit, [0,1] internal normalisation, Powell polish.
#
# Usage:
#   sbatch scripts/run_pso_global.sh                       # all defaults
#   sbatch scripts/run_pso_global.sh 500 50                # custom iter/swarm
#   sbatch scripts/run_pso_global.sh 500 50 yes            # + log-scale
#   sbatch scripts/run_pso_global.sh 500 50 yes 42         # + seed
#   sbatch scripts/run_pso_global.sh 500 50 yes 42 yes     # + skip polish
# ==============================================================================

#SBATCH --job-name=pso_global
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --output=pso_global_%j.out
#SBATCH --error=pso_global_%j.err

# ==============================================================================
# Arguments
# ==============================================================================
MAX_ITER=${1:-500}
SWARM_SIZE=${2:-50}
LOG_SCALE=${3:-no}        # "yes" to optimise selected params in log space
SEED=${4:-42}             # random seed for LHS + PSO
NO_POLISH=${5:-no}        # "yes" to skip local Powell polish

# ==============================================================================
# Environment
# ==============================================================================
set -e

echo "=================================================="
echo "Advanced PSO Global Optimisation"
echo "=================================================="
echo "Date:        $(date)"
echo "Node:        $(hostname)"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Max iter:    ${MAX_ITER}"
echo "Swarm size:  ${SWARM_SIZE}"
echo "Log scale:   ${LOG_SCALE}"
echo "Seed:        ${SEED}"
echo "No polish:   ${NO_POLISH}"
echo "=================================================="

cd /bigwork/nhkbarit/thesis-code/code

module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

# Let JAX use GPU if available; pyswarm runs on CPU anyway but
# the JIT-compiled simulation can benefit from GPU XLA.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORM_NAME=cpu 

# Build optional flags and choose output path
if [ "${LOG_SCALE}" = "yes" ]; then
    LOG_SCALE_FLAG="--log-scale"
    OUTPUT_PATH="notebooks/experiments/pso_global_log_optimization_results.json"
else
    LOG_SCALE_FLAG=""
    OUTPUT_PATH="notebooks/experiments/pso_global_optimization_results.json"
fi

if [ "${NO_POLISH}" = "yes" ]; then
    POLISH_FLAG="--no-polish"
else
    POLISH_FLAG=""
fi

# ==============================================================================
# Run
# ==============================================================================
srun python scripts/run_pso_global.py \
    --data-dir ../data/measurements \
    --max-iter ${MAX_ITER} \
    --swarm-size ${SWARM_SIZE} \
    --seed ${SEED} \
    --T-seg 800 \
    --output ${OUTPUT_PATH} \
    ${LOG_SCALE_FLAG} \
    ${POLISH_FLAG}

echo "=================================================="
echo "[$(date)] Completed"
echo "=================================================="
