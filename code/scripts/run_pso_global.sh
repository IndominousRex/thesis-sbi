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
#   sbatch scripts/run_pso_global.sh                              # all defaults
#   sbatch scripts/run_pso_global.sh 3000 200                     # custom iter/swarm
#   sbatch scripts/run_pso_global.sh 3000 200 yes                 # + log-scale
#   sbatch scripts/run_pso_global.sh 3000 200 yes 42              # + seed
#   sbatch scripts/run_pso_global.sh 3000 200 yes 42 yes          # + skip polish
#   sbatch scripts/run_pso_global.sh 3000 200 yes 42 no path/to/prev.json  # warm-start
#   sbatch scripts/run_pso_global.sh 3000 200 yes 42 no "" 30 0.5 0.5  # tire_weight=0.5 (default)
# $9=TIRE_WEIGHT: tire channel weight relative to physics (1.0). 0=exclude, 0.5=default, 1.0=equal
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
MAX_ITER=${1:-3000}
SWARM_SIZE=${2:-200}
LOG_SCALE=${3:-yes}       # "yes" to optimise selected params in log space
SEED=${4:-42}             # random seed for LHS + PSO
NO_POLISH=${5:-no}        # "yes" to skip local Powell polish
WARM_START=${6:-}         # path to previous results JSON for warm-start (empty = none)
STAGNATION_LIMIT=${7:-30}
REINIT_FRACTION=${8:-0.5}
TIRE_WEIGHT=${9:-0.5}     # tire channel weight: 0=exclude, 0.5=default, 1.0=equal to physics

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
echo "Warm-start:  ${WARM_START:-none}"
echo "Stag. limit: ${STAGNATION_LIMIT}"
echo "Reinit frac: ${REINIT_FRACTION}"
echo "Tire weight: ${TIRE_WEIGHT}"
echo "=================================================="

cd /bigwork/nhkbarit/thesis-code/code

module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

# Let JAX use GPU if available; pyswarm runs on CPU anyway but
# the JIT-compiled simulation can benefit from GPU XLA.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORM_NAME || true

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

if [ -n "${WARM_START}" ]; then
    WARM_START_FLAG="--warm-start ${WARM_START}"
else
    WARM_START_FLAG=""
fi

TIRE_FLAG="--tire-weight ${TIRE_WEIGHT}"

# ==============================================================================
# Run
# ==============================================================================
srun python scripts/run_pso_global.py \
    --data-dir ../data/measurements \
    --max-iter ${MAX_ITER} \
    --swarm-size ${SWARM_SIZE} \
    --seed ${SEED} \
    --T-seg 800 \
    --stagnation-limit ${STAGNATION_LIMIT} \
    --reinit-fraction ${REINIT_FRACTION} \
    --output ${OUTPUT_PATH} \
    ${LOG_SCALE_FLAG} \
    ${POLISH_FLAG} \
    ${WARM_START_FLAG} \
    ${TIRE_FLAG}


echo "=================================================="
echo "[$(date)] Completed"
echo "=================================================="
