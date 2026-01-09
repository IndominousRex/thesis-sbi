#!/bin/bash -l
# ==============================================================================
# Compare all methods on the same cached dataset
#
# This script ensures fair comparison by:
# 1. First generating and caching the dataset with NPE
# 2. Running NPSE and FNPE with --reuse-dataset to use the same data
#
# Usage:
#   sbatch run_comparison.sh comparison_v1 --num-sim 2000 --T-seg 1000
# ==============================================================================

#SBATCH --job-name=sbi_comparison
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --output=sbi_cmp_%j.out
#SBATCH --error=sbi_cmp_%j.err

set -e

# ==============================================================================
# Parse arguments
# ==============================================================================
EXP_NAME=${1:-comparison}
shift 1 2>/dev/null || shift $# 2>/dev/null
EXTRA_ARGS="$@"

echo "=================================================="
echo "[$(date)] SBI Method Comparison"
echo "Experiment: ${EXP_NAME}"
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
# Step 1: Run NPE (generates and caches dataset)
# ==============================================================================
echo ""
echo ">>> Running NPE (generating cached dataset)..."
echo ""
srun python run.py \
    --method npe \
    --exp-name "${EXP_NAME}_npe" \
    --device cuda \
    ${EXTRA_ARGS}

# ==============================================================================
# Step 2: Run NPSE with same dataset
# ==============================================================================
echo ""
echo ">>> Running NPSE (reusing cached dataset)..."
echo ""
srun python run.py \
    --method npse \
    --exp-name "${EXP_NAME}_npse" \
    --reuse-dataset \
    --device cuda \
    ${EXTRA_ARGS}

# ==============================================================================
# Step 3: Run FNPE with correct proposal-based training
# ==============================================================================
echo ""
echo ">>> Running FNPE with proposal-based training (correct implementation)..."
echo ""
srun python run.py \
    --method fnpe \
    --exp-name "${EXP_NAME}_fnpe" \
    --fnpe-proposal-type pred \
    --reuse-dataset \
    --device cuda \
    ${EXTRA_ARGS}

echo "=================================================="
echo "[$(date)] Comparison complete!"
echo "Results in: experiments/${EXP_NAME}_*/"
echo "=================================================="
echo "Results in: experiments/${EXP_NAME}_*/"
echo "=================================================="
