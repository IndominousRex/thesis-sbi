"""
Calibrated noise injection for bridging the sim-to-real gap.

Noise values were calibrated from 7 real measurement CSVs using
Savitzky-Golay residual analysis (see notebooks/realistic_simulation_improvements.ipynb).

Two noise layers:
  1. Process noise  – low-pass-filtered random walk on dynamic channels,
                      mimicking unmodeled dynamics (road surface, tire temp, etc.)
  2. Observation noise – i.i.d. Gaussian per channel, mimicking sensor noise.
"""

import numpy as np

# ─── Calibrated observation noise std (averaged over 7 real CSVs) ────────────
# Channel order: [yaw_rate, v_x, v_y, a_x, a_y, tire_fl, tire_fr, tire_rl, tire_rr]
OBS_NOISE_STD = np.array(
    [
        0.000046,  # yaw_rate   [rad/s]
        0.009000,  # v_body_x   [m/s]
        0.005000,  # v_body_y   [m/s]
        1.053000,  # a_body_x   [m/s²]
        0.430000,  # a_body_y   [m/s²]
        0.822000,  # tire_rate_fl [rad/s]
        0.822000,  # tire_rate_fr [rad/s]
        0.822000,  # tire_rate_rl [rad/s]
        0.822000,  # tire_rate_rr [rad/s]
    ],
    dtype=np.float32,
)

# ─── Process noise scales (random-walk std per time-step) ────────────────────
# These inject slowly-varying drift that mimics unmodeled dynamics.
PROCESS_NOISE_SCALES = {
    0: 0.0005,  # yaw_rate [rad/s]
    1: 0.005,  # v_x      [m/s]
    2: 0.002,  # v_y      [m/s]
    3: 0.01,  # a_x      [m/s²]
    4: 0.005,  # a_y      [m/s²]
    5: 0.02,  # tire_fl  [rad/s]
    6: 0.02,  # tire_fr  [rad/s]
    7: 0.02,  # tire_rl  [rad/s]
    8: 0.02,  # tire_rr  [rad/s]
}


def add_observation_noise(
    y_clean: np.ndarray,
    rng: np.random.Generator,
    noise_std: np.ndarray | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Add per-channel i.i.d. Gaussian sensor noise.

    Args:
        y_clean:   (T, D) clean observation array.
        rng:       numpy random Generator.
        noise_std: per-channel std array (D,).  Defaults to OBS_NOISE_STD.
        scale:     global multiplier on all noise (0.0 = off, 1.0 = calibrated).

    Returns:
        y_noisy: (T, D) noisy observation array.
    """
    if noise_std is None:
        noise_std = OBS_NOISE_STD
    noise = rng.normal(0.0, noise_std * scale, size=y_clean.shape).astype(np.float32)
    return y_clean + noise


def add_process_noise(
    y_clean: np.ndarray,
    rng: np.random.Generator,
    scales: dict | None = None,
    smoothing_window: int = 5,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Add low-pass-filtered random-walk perturbations to each dynamic channel.

    This mimics slowly-varying unmodeled dynamics (road surface changes,
    tire temperature drift, etc.) rather than white noise.

    Args:
        y_clean:          (T, D) clean observation array.
        rng:              numpy random Generator.
        scales:           dict mapping channel index -> per-step noise std.
                          Defaults to PROCESS_NOISE_SCALES.
        smoothing_window: moving-average kernel size for smoothing the walk.
        scale:            global multiplier (0.0 = off, 1.0 = calibrated).

    Returns:
        y_out: (T, D) perturbed observation array.
    """
    if scales is None:
        scales = PROCESS_NOISE_SCALES
    T, D = y_clean.shape
    y_out = y_clean.copy()
    for ch_idx, ch_scale in scales.items():
        if ch_idx >= D:
            continue
        increments = rng.normal(0.0, ch_scale * scale, size=T).astype(np.float32)
        walk = np.cumsum(increments)
        if smoothing_window > 1:
            kernel = np.ones(smoothing_window, dtype=np.float32) / smoothing_window
            walk = np.convolve(walk, kernel, mode="same").astype(np.float32)
        y_out[:, ch_idx] += walk
    return y_out


def add_sim_noise(
    y_clean: np.ndarray,
    rng: np.random.Generator,
    *,
    obs_noise_std: np.ndarray | None = None,
    process_noise_scales: dict | None = None,
    obs_noise_scale: float = 1.0,
    process_noise_scale: float = 1.0,
    smoothing_window: int = 5,
) -> np.ndarray:
    """
    Convenience function: apply process noise then observation noise.

    This is the main entry point used by the simulator.

    Args:
        y_clean:              (T, D) clean observation array.
        rng:                  numpy random Generator.
        obs_noise_std:        per-channel obs noise std (D,).
        process_noise_scales: dict of per-channel random-walk scales.
        obs_noise_scale:      global multiplier for obs noise (0 = off).
        process_noise_scale:  global multiplier for process noise (0 = off).
        smoothing_window:     moving-average window for process noise.

    Returns:
        y_noisy: (T, D) noisy observation array.
    """
    y = y_clean
    if process_noise_scale > 0:
        y = add_process_noise(
            y,
            rng,
            scales=process_noise_scales,
            smoothing_window=smoothing_window,
            scale=process_noise_scale,
        )
    if obs_noise_scale > 0:
        y = add_observation_noise(
            y, rng, noise_std=obs_noise_std, scale=obs_noise_scale
        )
    return y
