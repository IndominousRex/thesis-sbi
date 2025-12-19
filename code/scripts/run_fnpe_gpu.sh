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
# Params matched to NPE runs for fair comparison
# =============================================================================

echo "======================================================"
echo "[$(date)] Starting FNPE Vehicle Dynamics Experiment"
echo "======================================================"

EXP_NAME="fnpe_vehicle_allparams"
NUM_SIM=20000           
T_OBS=3000              
EPOCHS=100               
STEPS_PER_EPOCH=1000    
BATCH_SIZE=32           
HIDDEN_DIM=32           
NUM_HIDDEN=3            
MODEL_TYPE="gru"        
LR=1e-3                 
NUM_DIFF_STEPS=500      # Reverse diffusion steps
NUM_POST_SAMPLES=1000   
SEED=42                 
PARAMS="mu,cd,m"        

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
