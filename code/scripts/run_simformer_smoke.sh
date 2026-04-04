#!/bin/bash -l
#SBATCH --job-name=simf_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=simf_smoke_%j.out
#SBATCH --error=simf_smoke_%j.err

set -e
cd /bigwork/nhkbarit/thesis-code/code
module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORM_NAME || true

python - <<'PY'
from datetime import datetime
from configs.config import ExperimentConfig
from inference.unified_experiment import run_experiment

exp_name = f"simformer_smoke_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

cfg = ExperimentConfig(
    method="simformer",
    exp_name=exp_name,
    device="cuda",
    num_simulations=64,
    T_seg=128,
    cache_dataset=False,
    run_sbc=True,
    num_sbc_samples=16,
    run_one_step_rmse=True,
    run_posterior_plots=False,
    no_plots=True,
    real_data_csv=None,
    do_train=True,
    do_eval=True,
    simformer_train_steps_scaling=1,
    simformer_min_train_steps=200,
    simformer_max_train_steps=200,
    simformer_batch_size=64,
    simformer_num_layers=2,
    simformer_num_heads=2,
    simformer_attn_size=8,
    simformer_token_dim=16,
    simformer_num_diffusion_steps=50,
)

run_experiment(cfg)
PY
