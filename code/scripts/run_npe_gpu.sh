#!/bin/bash -l
#SBATCH --job-name=npe_paramsweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=npe_gpu_%A_%a.out
#SBATCH --error=npe_gpu_%A_%a.err
#SBATCH --array=0-3

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

echo "[$(date)] Starting array task ${IDX} with params='${PARAMS}' (suffix='${EXP_SUFFIX}')"

cd /bigwork/nhkbarit/thesis-code/code

module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=cpu

# -------------------
# RUN THE EXPERIMENT
# -------------------

srun python main.py \
    --exp-name baseline_bigru_maf_${EXP_SUFFIX} \
    --num-sim 20000 \
    --device cuda \
    --seed 42 \
    --dt 0.01 \
    --T-seg 3000 \
    --decimate 2 \
    --encoder bigru \
    --encoder-hidden 32 \
    --lr 1e-3 \
    --batch-train 32 \
    --stop-after-epochs 20 \
    --params "${PARAMS}" \
    --real-data-csv "../data/measurements/Jeversen_2022_10_12_110132.csv"
