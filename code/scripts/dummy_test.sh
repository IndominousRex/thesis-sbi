#!/bin/bash -l
#SBATCH --job-name=npe_dummy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=npe_allparams_%j.out
#SBATCH --error=npe_allparams_%j.err

echo "=== Running DUMMY end-to-end test ==="

# --- go to code directory ---
cd /bigwork/nhkbarit/thesis-code/code

# --- load env ---
module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=cpu

# -----------------------------
# 1) TINY TRAINING RUN
# -----------------------------
echo "[Dummy] Running tiny training run..."

srun python main.py \
    --exp-name dummy_test \
    --num-sim 200 \
    --device cuda \
    --seed 42 \
    --dt 0.01 \
    --T-seg 200 \
    --decimate 2 \
    --encoder transformer \
    --lr 1e-3 \
    --batch-train 32 \
    --stop-after-epochs 1 \
    --num-lc2st-samples 100 \
    --params "mu,cd,m" \
    --real-data-csv "../data/measurements/Jeversen_2022_10_12_110132.csv"

echo "=== Dummy test completed ==="