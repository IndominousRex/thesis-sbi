#!/bin/bash -l
#SBATCH --job-name=fnpe_paramsweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=08:00:00
#SBATCH --output=fnpe_paramsweep_%A_%a.out
#SBATCH --error=fnpe_paramsweep_%A_%a.err
#SBATCH --array=0-3

# =============================================================================
# FNPE Parameter Sweep Script
# Runs FNPE with different parameter subsets (similar to NPE param sweep)
# =============================================================================

# -------------
# PARAMETER SETS
# -------------
# 0 -> infer mu only
# 1 -> infer cd only
# 2 -> infer m only
# 3 -> infer all three (mu, cd, m)

PARAMS_LIST=("mu" "cd" "m" "mu,cd,m")
EXP_SUFFIX_LIST=("mu_only" "cd_only" "m_only" "all_three")

IDX=${SLURM_ARRAY_TASK_ID}
PARAMS="${PARAMS_LIST[$IDX]}"
EXP_SUFFIX="${EXP_SUFFIX_LIST[$IDX]}"

echo "======================================================"
echo "[$(date)] Starting FNPE array task ${IDX}"
echo "  Parameters: ${PARAMS}"
echo "  Experiment suffix: ${EXP_SUFFIX}"
echo "======================================================"

# --- Common configuration ---
NUM_SIM=100000
T_OBS=100
EPOCHS=25
STEPS_PER_EPOCH=10000
BATCH_SIZE=256
HIDDEN_DIM=128
NUM_HIDDEN=5
MODEL_TYPE="gru"
LR=5e-4
NUM_DIFF_STEPS=500
SEED=42

# --- Go to code directory ---
cd /bigwork/nhkbarit/thesis-code/code

# --- Load environment ---
module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

# --- JAX configuration ---
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=cpu

echo ""
echo "[CONFIG]"
echo "  Num simulations: ${NUM_SIM}"
echo "  T_obs: ${T_OBS}"
echo "  Epochs: ${EPOCHS}"
echo "  Steps per epoch: ${STEPS_PER_EPOCH}"
echo ""

# --- GPU status ---
echo "[GPU] Before run:"
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free --format=csv
echo ""

# --- Run FNPE experiment ---
srun python -m inference.fnpe_experiment \
    --exp-name "fnpe_${EXP_SUFFIX}" \
    --num-sim ${NUM_SIM} \
    --T-obs ${T_OBS} \
    --epochs ${EPOCHS} \
    --steps-per-epoch ${STEPS_PER_EPOCH} \
    --batch-size ${BATCH_SIZE} \
    --hidden-dim ${HIDDEN_DIM} \
    --num-hidden ${NUM_HIDDEN} \
    --model-type ${MODEL_TYPE} \
    --lr ${LR} \
    --num-diff-steps ${NUM_DIFF_STEPS} \
    --seed ${SEED} \
    --params "${PARAMS}"

EXIT_CODE=$?

echo ""
echo "[GPU] After run:"
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free --format=csv
echo ""

echo "======================================================"
echo "[$(date)] FNPE task ${IDX} finished with exit code ${EXIT_CODE}"
echo "======================================================"

exit ${EXIT_CODE}
