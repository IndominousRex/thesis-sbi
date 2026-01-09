#!/bin/bash -l
# ==============================================================================
# Unified SLURM script for SBI experiments
#
# Supports NPE, NPSE, and FNPE with identical data generation for fair comparisons.
#
# Usage examples:
#   sbatch run_experiment.sh npe baseline
#   sbatch run_experiment.sh npse baseline --sde-type ve
#   sbatch run_experiment.sh fnpe baseline --fnpe-proposal-type pred
#   sbatch run_experiment.sh fnpe old_impl --fnpe-proposal-type trajectory
#   sbatch --array=0-2 run_experiment.sh sweep_methods  # Run all methods
#
# FNPE Proposal Types:
#   pred       - (DEFAULT) Correct implementation per FNPE paper
#   trajectory - Old implementation for comparison/ablation
#   naive      - Simple expanded prior (for testing)
# ==============================================================================

#SBATCH --job-name=sbi_experiment
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=sbi_exp_%j.out
#SBATCH --error=sbi_exp_%j.err

# ==============================================================================
# Parse arguments
# ==============================================================================
METHOD=${1:-npe}
EXP_NAME=${2:-baseline}
shift 2 2>/dev/null || shift $# 2>/dev/null
EXTRA_ARGS="$@"

# ==============================================================================
# Environment setup
# ==============================================================================
set -e

echo "=================================================="
echo "SBI Experiment: ${METHOD} / ${EXP_NAME}"
echo "=================================================="
echo "Date:      $(date)"
echo "Node:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Method:    ${METHOD}"
echo "Exp Name:  ${EXP_NAME}"
echo "Extra Args: ${EXTRA_ARGS}"
echo "=================================================="

# --- Go to code directory ---
cd /bigwork/nhkbarit/thesis-code/code

# --- Load environment ---
module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

# --- JAX configuration (use CPU) ---
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=cpu

# ==============================================================================
# Run experiment
# ==============================================================================
srun python run.py \
    --method ${METHOD} \
    --exp-name ${EXP_NAME} \
    --device cuda \
    ${EXTRA_ARGS}

echo "=================================================="
echo "[$(date)] Completed"
echo "=================================================="
