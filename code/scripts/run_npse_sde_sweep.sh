#!/bin/bash -l
#SBATCH --job-name=npse_sde_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --output=npse_sde_%A_%a.out
#SBATCH --error=npse_sde_%A_%a.err
#SBATCH --array=0-2

# NPSE SDE type sweep experiment
# Compares Variance Exploding (ve), Variance Preserving (vp), and sub-VP (subvp)

# -------------
# SDE TYPES
# -------------

SDE_LIST=("ve" "vp" "subvp")

IDX=${SLURM_ARRAY_TASK_ID}
SDE_TYPE="${SDE_LIST[$IDX]}"

echo "[$(date)] Starting NPSE task ${IDX} with sde_type='${SDE_TYPE}'"

cd /bigwork/nhkbarit/thesis-code/code

module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=cpu

# -------------------
# RUN THE NPSE EXPERIMENT
# -------------------

srun python main_npse.py \
    --exp-name baseline_bigru_npse_${SDE_TYPE} \
    --num-sim 20000 \
    --device cuda \
    --seed 42 \
    --dt 0.01 \
    --T-seg 3000 \
    --encoder-type bigru \
    --encoder-hidden 32 \
    --lr 1e-3 \
    --batch-train 512 \
    --stop-after-epochs 20 \
    --params "mu,cd,m" \
    --sde-type "${SDE_TYPE}"

echo "[$(date)] NPSE task ${IDX} (sde_type=${SDE_TYPE}) finished."
