#!/bin/bash -l
#SBATCH --job-name=npe_vehicle
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --output=npe_gpu_%j.out
#SBATCH --error=npe_gpu_%j.err
#SBATCH --mail-user=your.mail@stud.uni-hannover.de
#SBATCH --mail-type=END,FAIL

# go back to the folder where you submitted the job from
cd "../"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# load conda
module load Miniforge3   # adapt if your cluster uses a different module name

# activate the env in BIGWORK
conda activate /bigwork/nhkbarit/conda_envs/npe

# running the experiment
srun python main.py \
    --exp-name baseline_bigru_maf \
    --num-sim 2000 \
    --device cuda \
    --seed 42 \
    --dt 0.01 \
    --T-seg 3000 \
    --decimate 2 \
    --encoder bigru \
    --lr 1e-3 \
    --batch-train 512 \
    --stop-after-epochs 20

EOF