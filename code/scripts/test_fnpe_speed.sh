#!/bin/bash
#SBATCH --job-name=fnpe_speed_test
#SBATCH --output=logs/fnpe_speed_test_%j.out
#SBATCH --error=logs/fnpe_speed_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# ==============================================================================
# FNPE Speed Test - Quick benchmark to verify training performance
# Usage: sbatch scripts/test_fnpe_speed.sh
# ==============================================================================

echo "=================================================="
echo "FNPE Speed Test"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "=================================================="

# --- Setup ---
cd /bigwork/nhkbarit/thesis-code/code
mkdir -p logs

# --- Load environment ---
module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

# --- JAX configuration (use CPU) ---
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=cpu

# ==============================================================================
# Run quick FNPE test (3 epochs, 1000 steps each)
# ==============================================================================
echo ""
echo ">>> Running FNPE with window_size=2 (FIXED - was using T_seg before!)"
echo ""

srun python -c "
import sys
import time
sys.path.insert(0, '.')

from utils.env_utils import setup_environment
setup_environment(42)

import jax
print(f'JAX devices: {jax.devices()}')
print(f'JAX backend: {jax.default_backend()}')

from configs.config import ExperimentConfig
from methods.fnpe_method import FNPEMethod

# Test config
cfg = ExperimentConfig(
    num_simulations=10000,
    T_seg=100,
    active_parameters=['mu', 'Cd', 'm'],
    obs_dim=9,
)

print()
print('=' * 60)
print('FNPE Speed Test')
print('=' * 60)
print(f'Simulations: {cfg.num_simulations}')
print(f'T_seg: {cfg.T_seg}')
print(f'Parameters: {cfg.active_parameters}')

# Create method with small window_size (THE FIX!)
method = FNPEMethod(
    cfg=cfg,
    prior=None,
    device='cpu',
    hidden_dim=128,
    num_hidden=5,
    model_type='gru',
    window_size=2,  # CRITICAL: small window like Lotka-Volterra!
    num_epochs=3,
    steps_per_epoch=1000,
    batch_size=512,
    stop_after_epochs=10,
    validation_fraction=0.1,
)

print(f'Window size: {method.window_size} (was using T_seg={cfg.T_seg} before fix!)')

# Build
print()
print('[TEST] Building method...')
t0 = time.time()
method.build(input_dim=cfg.obs_dim, seq_len=cfg.T_seg)
print(f'[TEST] Build time: {time.time() - t0:.2f}s')

# Train
print()
print('[TEST] Training (3 epochs, 1000 steps each = 3000 steps)...')
t0 = time.time()
summary = method.train(num_simulations=cfg.num_simulations, T_obs=cfg.T_seg)
train_time = time.time() - t0

print()
print('=' * 60)
print('RESULTS')
print('=' * 60)
print(f'Total train time: {train_time:.2f}s')
print(f'Time per epoch: {train_time / 3:.2f}s')
print(f'Time per step: {train_time / 3000 * 1000:.2f}ms')
print(f'Final train loss: {summary.get(\"final_train_loss\", \"N/A\")}')
print(f'Final val loss: {summary.get(\"final_val_loss\", \"N/A\")}')

# Extrapolate
print()
print('[ESTIMATE] For full run (20 epochs, 10000 steps/epoch = 200k steps):')
estimated_time = (train_time / 3000) * 200000
print(f'  Estimated time: {estimated_time/3600:.1f} hours')
print(f'  (vs ~12+ hours before the fix!)')
"

echo ""
echo "=================================================="
echo "[$(date)] Completed"
echo "=================================================="
