#!/bin/bash -l
# ==============================================================================
# SLURM script for PSO optimisation with global tire params
#
# Optimises all vehicle/tire parameters jointly across 7 trajectories.
# Only mu (friction) varies per trajectory. Total: 23 parameters.
#
# Usage:
#   sbatch scripts/run_pso_global.sh              # defaults (500 iter)
#   sbatch scripts/run_pso_global.sh 500 50       # custom iter / swarm
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

# ==============================================================================
# Environment
# ==============================================================================
set -e

echo "=================================================="
echo "PSO Global Optimisation"
echo "=================================================="
echo "Date:        $(date)"
echo "Node:        $(hostname)"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Max iter:    ${MAX_ITER}"
echo "Swarm size:  ${SWARM_SIZE}"
echo "=================================================="

cd /bigwork/nhkbarit/thesis-code/code

module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

# Let JAX use GPU if available; pyswarm runs on CPU anyway but
# the JIT-compiled simulation can benefit from GPU XLA.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORM_NAME=cpu 

# ==============================================================================
# Run
# ==============================================================================
srun python scripts/run_pso_global.py \
    --data-dir ../data/measurements \
    --max-iter ${MAX_ITER} \
    --swarm-size ${SWARM_SIZE} \
    --T-seg 800 \
    --output notebooks/experiments/pso_global_optimization_results.json

echo "=================================================="
echo "[$(date)] Completed"
echo "=================================================="
