#!/bin/bash
#SBATCH --job-name=npe_dummy_archs
#SBATCH --output=npe_dummy_%A_%a.out
#SBATCH --error=npe_dummy_%A_%a.err
#SBATCH --array=0-2              # 0 = bigru, 1 = transformer, 2 = causalcnn
#SBATCH --time=01:00:00          
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1     
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd /bigwork/nhkbarit/thesis-code/code

module load Miniforge3/25.3.0-3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=cpu

# -------------------------
# ARCHITECTURE SELECTION
# -------------------------
ARCHS=("bigru" "transformer" "causalcnn")
ENCODER="${ARCHS[$SLURM_ARRAY_TASK_ID]}"

echo "Running dummy test for encoder: $ENCODER"

# -------------------------
# RUN THE DUMMY EXPERIMENT
# -------------------------
srun python main.py \
    --exp-name dummy_${ENCODER} \
    --num-sim 200 \
    --device cuda \
    --seed 42 \
    --dt 0.01 \
    --T-seg 200 \
    --decimate 2 \
    --encoder-type $ENCODER \
    --encoder-hidden 32 \
    --lr 1e-3 \
    --batch-train 32 \
    --stop-after-epochs 1 \
    --num-lc2st-samples 20 \
    --params "mu,cd,m" \
    --real-data-csv "../data/measurements/Jeversen_2022_10_12_110132.csv"

echo "Dummy run for $ENCODER completed."
