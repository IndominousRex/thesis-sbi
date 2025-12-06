#!/bin/bash
#SBATCH --job-name=npe_archs_allparams
#SBATCH --array=0-2              # 0 = bigru, 1 = transformer, 2 = causalcnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1    
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00
#SBATCH --output=npe_archs_allparams_%j.out
#SBATCH --error=npe_archs_allparams_%j.err

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
# RUN THE BASELINE EXPERIMENT
# -------------------------
srun python main.py \
    --exp-name baseline_${ENCODER}_allparams \
    --num-sim 20000 \
    --device cuda \
    --seed 42 \
    --dt 0.01 \
    --T-seg 3000 \
    --encoder-type $ENCODER \
    --encoder-hidden 32 \
    --lr 1e-3 \
    --batch-train 256 \
    --stop-after-epochs 20 \
    --num-lc2st-samples 1000 \
    --params "mu,cd,m" \
    --real-data-csv "../data/measurements/Jeversen_2022_10_12_110132.csv"
