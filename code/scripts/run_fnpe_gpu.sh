#!/bin/bash -l
#SBATCH --job-name=fnpe_vehicle
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --output=fnpe_gpu_%j.out
#SBATCH --error=fnpe_gpu_%j.err

# =============================================================================
# FNPE (Flow-based Neural Posterior Estimation) Experiment Script
# Uses the MarkovSBI framework with diffusion-based score matching
# =============================================================================

echo "======================================================"
echo "[$(date)] Starting FNPE Vehicle Dynamics Experiment"
echo "======================================================"

# --- Configuration ---
EXP_NAME="fnpe_vehicle_allparams"
NUM_SIM=100000          # More data typically helps FNPE
T_OBS=100               # Trajectory length (markovsbi style, shorter than NPE)
EPOCHS=25               # Number of training epochs
STEPS_PER_EPOCH=10000   # Gradient steps per epoch
BATCH_SIZE=256          # Training batch size
HIDDEN_DIM=128          # Score network hidden dim
NUM_HIDDEN=5            # Score network depth
MODEL_TYPE="gru"        # Score network observation processing (gru or linear)
LR=5e-4                 # Learning rate
NUM_DIFF_STEPS=500      # Reverse diffusion steps
NUM_POST_SAMPLES=1000   # Posterior samples for diagnostics
SEED=42
PARAMS="mu,cd,m"        # Parameters to infer

# --- Go to code directory ---
cd /bigwork/nhkbarit/thesis-code/code

# --- Load environment ---
module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

# --- JAX configuration ---
# Use GPU for JAX operations
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=cpu

echo ""
echo "[CONFIG]"
echo "  Experiment name: ${EXP_NAME}"
echo "  Num simulations: ${NUM_SIM}"
echo "  Trajectory length (T_obs): ${T_OBS}"
echo "  Epochs: ${EPOCHS}"
echo "  Steps per epoch: ${STEPS_PER_EPOCH}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Hidden dim: ${HIDDEN_DIM}"
echo "  Num hidden layers: ${NUM_HIDDEN}"
echo "  Model type: ${MODEL_TYPE}"
echo "  Learning rate: ${LR}"
echo "  Diffusion steps: ${NUM_DIFF_STEPS}"
echo "  Posterior samples: ${NUM_POST_SAMPLES}"
echo "  Parameters: ${PARAMS}"
echo "  Seed: ${SEED}"
echo ""

# --- GPU status ---
echo "[GPU] Before run:"
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free --format=csv
echo ""

# --- Run FNPE experiment ---
srun python -m inference.fnpe_experiment \
    --exp-name "${EXP_NAME}" \
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
    --num-post-samples ${NUM_POST_SAMPLES} \
    --seed ${SEED} \
    --params "${PARAMS}"

EXIT_CODE=$?

echo ""
echo "[GPU] After run:"
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free --format=csv
echo ""

echo "======================================================"
echo "[$(date)] FNPE experiment finished with exit code ${EXIT_CODE}"
echo "======================================================"

exit ${EXIT_CODE}
