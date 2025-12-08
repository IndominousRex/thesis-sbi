#!/bin/bash -l
#SBATCH --job-name=npe_allparams
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00
#SBATCH --output=npe_allparams_%j.out
#SBATCH --error=npe_allparams_%j.err

echo "[$(date)] Starting joint-parameter run (mu, cd, m)"

# --- go to code directory ---
cd /bigwork/nhkbarit/thesis-code/code

# --- load env ---
module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=cpu

# --- run experiment: infer all three parameters ---
srun python main.py \
    --exp-name baseline_transformer_maf_all_three \
    --num-sim 10000 \
    --device cuda \
    --seed 42 \
    --dt 0.01 \
    --T-seg 3000 \
    --encoder-type transformer \
    --encoder-hidden 32 \
    --lr 1e-3 \
    --batch-train 256 \
    --stop-after-epochs 20 \
    --params "mu,cd,m" \
    --real-data-csv "../data/measurements/Jeversen_2022_10_12_110132.csv"
