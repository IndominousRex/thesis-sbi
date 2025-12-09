#!/bin/bash -l
#SBATCH --job-name=npe_transformer_tseg_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00
#SBATCH --output=npe_transformer_tseg_sweep_%j.out
#SBATCH --error=npe_transformer_tseg_sweep_%j.err

echo "[$(date)] Starting Transformer T_seg sweep"

# --- go to code directory ---
cd /bigwork/nhkbarit/thesis-code/code

# --- load env ---
module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe


export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=cpu

# ---- LIST OF T_SEG VALUES TO TEST ----
TSEGS=(300 500 1000 1500 2000)

log_gpu() {
    echo "------ GPU MEMORY STATUS ($(date)) ------"
    nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free --format=csv
    echo "-----------------------------------------"
    echo ""
}

for T in "${TSEGS[@]}"; do
    echo "======================================================"
    echo "[$(date)] Running Transformer experiment with T_seg = $T"
    echo "======================================================"

    echo "[GPU] BEFORE RUN (T_seg=$T):"
    log_gpu

    srun python main.py \
        --exp-name transformer_tseg_${T} \
        --num-sim 20000 \
        --device cuda \
        --seed 42 \
        --dt 0.01 \
        --T-seg ${T} \
        --encoder-type transformer \
        --encoder-hidden 32 \
        --lr 1e-3 \
        --batch-train 128 \
        --stop-after-epochs 20 \
        --params "mu,cd,m" \
        --real-data-csv "../data/measurements/Jeversen_2022_10_12_110132.csv"

    echo "[GPU] AFTER RUN (T_seg=$T):"
    log_gpu
    
    echo "[$(date)] Finished T_seg=$T"
    echo ""

done

echo "[$(date)] All runs completed."
