#!/bin/bash -l
# ==============================================================================
# Quick test script to verify all methods work
# Uses minimal settings for fast execution
# ==============================================================================

#SBATCH --job-name=test_all_methods
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=test_all_%j.out
#SBATCH --error=test_all_%j.err

set -e

echo "=================================================="
echo "[$(date)] Testing all SBI methods"
echo "=================================================="

# --- Go to code directory ---
cd /bigwork/nhkbarit/thesis-code/code

# --- Load environment ---
module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

# --- JAX configuration ---
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORM_NAME || true

# Real data path
REAL_DATA_CSV="../data/measurements/Jeversen_2022_10_12_110132.csv"

# Common parameters for fair comparison
COMMON_ARGS="--num-sim 100 --T-seg 300 --num-epochs 2 --stop-after-epochs 2 --num-sbc-samples 10 --no-swd --no-one-step --device cuda --real-data-csv $REAL_DATA_CSV"

echo ""
echo ">>> [1/3] Testing NPE..."
echo ""
srun python run.py --method npe --exp-name test_npe $COMMON_ARGS

echo ""
echo ">>> [2/3] Testing NPSE..."
echo ""
srun python run.py --method npse --exp-name test_npse --sde-type ve $COMMON_ARGS

echo ""
echo ">>> [3/3] Testing FNPE (with proposal-based training)..."
echo ""
srun python run.py --method fnpe --exp-name test_fnpe --fnpe-proposal-type pred --fnpe-steps-per-epoch 100 $COMMON_ARGS

echo ""
echo "=================================================="
echo "[$(date)] All methods tested successfully!"
echo "Results in: experiments/test_*/"
echo "=================================================="
