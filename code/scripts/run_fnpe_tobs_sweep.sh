#!/bin/bash -l
#SBATCH --job-name=fnpe_tobs_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=fnpe_tobs_sweep_%j.out
#SBATCH --error=fnpe_tobs_sweep_%j.err

# =============================================================================
# FNPE T_obs (trajectory length) Sweep Script
# Tests how trajectory length affects FNPE performance
# =============================================================================

echo "======================================================"
echo "[$(date)] Starting FNPE T_obs sweep"
echo "======================================================"

# --- Go to code directory ---
cd /bigwork/nhkbarit/thesis-code/code

# --- Load environment ---
module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

# --- JAX configuration ---
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=gpu

# ---- LIST OF T_OBS VALUES TO TEST ----
# Note: FNPE typically uses shorter sequences than NPE
TOBS_VALUES=(50 100 200 500 1000)

# --- Common configuration ---
NUM_SIM=100000
EPOCHS=25
STEPS_PER_EPOCH=10000
BATCH_SIZE=256
HIDDEN_DIM=128
NUM_HIDDEN=5
MODEL_TYPE="gru"
LR=5e-4
NUM_DIFF_STEPS=500
SEED=42
PARAMS="mu,cd,m"

log_gpu() {
    echo "------ GPU MEMORY STATUS ($(date)) ------"
    nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free --format=csv
    echo "-----------------------------------------"
    echo ""
}

for T in "${TOBS_VALUES[@]}"; do
    echo "======================================================"
    echo "[$(date)] Running FNPE experiment with T_obs = $T"
    echo "======================================================"

    echo "[GPU] BEFORE RUN (T_obs=$T):"
    log_gpu

    srun python -m inference.fnpe_experiment \
        --exp-name "fnpe_tobs_${T}" \
        --num-sim ${NUM_SIM} \
        --T-obs ${T} \
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

    echo "[GPU] AFTER RUN (T_obs=$T):"
    log_gpu

    echo "[$(date)] T_obs=$T finished with exit code $EXIT_CODE"
    echo ""

    # Small delay between runs
    sleep 5
done

echo "======================================================"
echo "[$(date)] FNPE T_obs sweep completed"
echo "======================================================"
