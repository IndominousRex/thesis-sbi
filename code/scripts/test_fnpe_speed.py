#!/usr/bin/env python
"""
Quick FNPE speed test - runs a few epochs with minimal data to benchmark training.
Usage: python scripts/test_fnpe_speed.py
"""

import sys
import time

sys.path.insert(0, ".")

from utils.env_utils import setup_environment

setup_environment(42)

import jax

print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

from configs.config import ExperimentConfig
from methods.fnpe_method import FNPEMethod

# Minimal config for speed test
cfg = ExperimentConfig(
    num_simulations=10000,  # Like Lotka-Volterra uses large dataset
    T_seg=100,  # Sequence length (doesn't affect training speed much now!)
    active_parameters=["mu", "Cd", "m"],  # 3 params
    obs_dim=9,
)

print("\n" + "=" * 60)
print("FNPE Speed Test")
print("=" * 60)
print(f"Simulations: {cfg.num_simulations}")
print(f"T_seg: {cfg.T_seg}")
print(f"Parameters: {cfg.active_parameters}")

# Create method - KEY: window_size=2 (small Markov window!)
method = FNPEMethod(
    cfg=cfg,
    prior=None,  # FNPE doesn't use torch prior
    device="cpu",
    hidden_dim=128,
    num_hidden=5,
    model_type="gru",
    window_size=2,  # CRITICAL: small window like Lotka-Volterra!
    num_epochs=3,  # Just 3 epochs
    steps_per_epoch=1000,  # 1000 steps per epoch
    batch_size=512,
    stop_after_epochs=10,
    validation_fraction=0.1,
)

print(f"Window size: {method.window_size} (CRITICAL - was using T_seg before!)")

# Build
print("\n[TEST] Building method...")
t0 = time.time()
method.build(input_dim=cfg.obs_dim, seq_len=cfg.T_seg)
print(f"[TEST] Build time: {time.time() - t0:.2f}s")

# Train
print("\n[TEST] Training (3 epochs, 1000 steps each)...")
t0 = time.time()
summary = method.train(num_simulations=cfg.num_simulations, T_obs=cfg.T_seg)
train_time = time.time() - t0

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Total train time: {train_time:.2f}s")
print(f"Time per epoch: {train_time / 3:.2f}s")
print(f"Time per step: {train_time / 3000 * 1000:.2f}ms")
print(f"Final train loss: {summary.get('final_train_loss', 'N/A')}")
print(f"Final val loss: {summary.get('final_val_loss', 'N/A')}")

# Extrapolate for full run
print("\n[ESTIMATE] For full run (20 epochs, 10000 steps/epoch = 200k steps):")
estimated_time = (train_time / 3000) * 200000
print(f"  Estimated time: {estimated_time/3600:.1f} hours")
print(f"  (vs ~12+ hours before the fix!)")
