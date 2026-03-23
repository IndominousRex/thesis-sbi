#!/bin/bash -l
# ==============================================================================
# FNPE Ablation Study: Compare proposal types
#
# This script compares the three FNPE proposal implementations:
# 1. pred (correct)   - Proposal from pilot simulation pool
# 2. trajectory (old) - Divide trajectories into pairs (biased)
# 3. naive           - Sample from initial state distribution only
#
# The "pred" implementation should show better mass parameter identification
# because it covers the full state space including high-velocity regimes.
#
# Usage:
#   sbatch scripts/run_fnpe_ablation.sh ablation_v1
#   sbatch scripts/run_fnpe_ablation.sh ablation_v1 --num-sim 5000 --T-seg 2000
# ==============================================================================

#SBATCH --job-name=fnpe_ablation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=fnpe_ablation_%j.out
#SBATCH --error=fnpe_ablation_%j.err

set -e

# ==============================================================================
# Parse arguments
# ==============================================================================
EXP_NAME=${1:-fnpe_ablation}
shift 1 2>/dev/null || shift $# 2>/dev/null
EXTRA_ARGS="$@"

echo "=================================================="
echo "[$(date)] FNPE Proposal Type Ablation Study"
echo "Experiment: ${EXP_NAME}"
echo "Extra Args: ${EXTRA_ARGS}"
echo "=================================================="
echo ""
echo "Comparing proposal types:"
echo "  - pred (correct): Sample states from pilot simulation pool"
echo "  - trajectory (old): Divide trajectories into pairs (BIASED)"
echo "  - naive: Sample from initial state distribution only"
echo ""

# --- Go to code directory ---
cd /bigwork/nhkbarit/thesis-code/code

# --- Load environment ---
module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

# --- JAX configuration ---
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORM_NAME || true

# Real data for evaluation
REAL_DATA="--real-data-csv ../data/measurements/Jeversen_2022_10_12_110132.csv"

# ==============================================================================
# Run 1: FNPE with pred proposal (CORRECT implementation)
# ==============================================================================
echo ""
echo ">>> [1/3] Running FNPE with 'pred' proposal (CORRECT implementation)..."
echo "    Training data: States sampled from full reachable distribution"
echo ""

srun python run.py \
    --method fnpe \
    --exp-name "${EXP_NAME}_pred" \
    --fnpe-proposal-type pred \
    --device cuda \
    ${REAL_DATA} \
    ${EXTRA_ARGS}

# ==============================================================================
# Run 2: FNPE with trajectory proposal (OLD/INCORRECT implementation)
# ==============================================================================
echo ""
echo ">>> [2/3] Running FNPE with 'trajectory' proposal (OLD implementation)..."
echo "    Training data: Consecutive pairs from trajectories (BIASED)"
echo ""

srun python run.py \
    --method fnpe \
    --exp-name "${EXP_NAME}_trajectory" \
    --fnpe-proposal-type trajectory \
    --device cuda \
    ${REAL_DATA} \
    ${EXTRA_ARGS}

# ==============================================================================
# Run 3: FNPE with naive proposal (baseline)
# ==============================================================================
echo ""
echo ">>> [3/3] Running FNPE with 'naive' proposal (baseline)..."
echo "    Training data: States from initial condition only"
echo ""

srun python run.py \
    --method fnpe \
    --exp-name "${EXP_NAME}_naive" \
    --fnpe-proposal-type naive \
    --device cuda \
    ${REAL_DATA} \
    ${EXTRA_ARGS}

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=================================================="
echo "[$(date)] FNPE Ablation Study Complete!"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  experiments/${EXP_NAME}_pred/       <- CORRECT implementation"
echo "  experiments/${EXP_NAME}_trajectory/ <- OLD implementation"
echo "  experiments/${EXP_NAME}_naive/      <- Baseline"
echo ""
echo "Expected findings:"
echo "  - 'pred' should have better mass parameter identification"
echo "  - 'trajectory' may show bias in SBC for mass"
echo "  - 'naive' should be worst (limited state coverage)"
echo "=================================================="
