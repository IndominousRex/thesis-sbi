#!/usr/bin/env python
"""
PSO optimization of vehicle model parameters (Approach 2b).

All vehicle/tire parameters are GLOBAL (shared across trajectories).
Only mu (friction coefficient) varies per trajectory.

Layout of the optimisation vector x:
  x[0 : N_GLOBAL]                          — 16 global vehicle/tire params
  x[N_GLOBAL : N_GLOBAL + N_TRAJECTORIES]  — per-trajectory mu values

Total: 16 + 7 = 23 parameters  (down from 47 in the original notebook).

Usage:
    python scripts/run_pso_global.py                              # local CPU
    python scripts/run_pso_global.py --data-dir /path/to/csvs     # custom data
    srun python scripts/run_pso_global.py --max-iter 500          # SLURM
"""

import sys
import os
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# JAX — prefer GPU if available, fall back to CPU
# ---------------------------------------------------------------------------
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

# Check device
_jax_devices = jax.devices()
print(f"[JAX] Available devices: {_jax_devices}")
print(f"[JAX] Default backend : {jax.default_backend()}")

from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Ensure project root is on the path so we can import simulation.*
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # code/
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.VehicleModel import (
    vehicle_RK4x,
    vehicle_fy,
    width_front,
    width_rear,
    length_front,
    length_rear,
    height_CoG,
    vel_limit,
    gravitation,
    air_density,
    body_surface_x,
)


# ============================================================================
# Parameter definitions
# ============================================================================

# --- Global params (shared across ALL trajectories) ---
GLOBAL_PARAMS = [
    "mass",
    "Inertia_z",
    "Inertia_tire",
    "Inertia_engine",
    "air_resistance",
    "c_1y",
    "c_2y",
    "C_y",
    "E_y",
    "C_roll1",
    "C_roll2",
    "radius_tire",
    # NEW: previously per-trajectory, now global
    "c_1x",
    "c_2x",
    "C_x",
    "E_x",
]

# --- Per-trajectory params (only mu) ---
PER_TRAJ_PARAMS = ["mu"]

NUM_GLOBAL = len(GLOBAL_PARAMS)  # 16
NUM_PER_TRAJ = len(PER_TRAJ_PARAMS)  # 1

# --- Default values ---
DEFAULTS = {
    "mass": 1720,
    "Inertia_z": 2066,
    "Inertia_tire": 28.6,
    "Inertia_engine": 0.197,
    "air_resistance": 0.27,
    "c_1y": 1.9e6,
    "c_2y": 1.35e5,
    "C_y": 1.9,
    "E_y": 0.52,
    "C_roll1": 0.0083,
    "C_roll2": 0.0005,
    "radius_tire": 0.3116,
    "c_1x": 2.5e7,
    "c_2x": 3.6e6,
    "C_x": 1.42,
    "E_x": -9.75,
    "mu": 0.8,
}

# --- Bounds (relaxed again: 10/16 params and mu still stuck at boundaries after second run) ---
GLOBAL_BOUNDS = {
    "mass": (800, 4000),
    "Inertia_z": (100, 6000),
    "Inertia_tire": (1, 200),
    "Inertia_engine": (0.01, 1.0),
    "air_resistance": (0.1, 5.0),
    "c_1y": (5e5, 2e7),
    "c_2y": (1e4, 1e7),
    "C_y": (0.5, 10.0),
    "E_y": (-6.0, 3.0),
    "C_roll1": (0.0005, 0.1),
    "C_roll2": (1e-7, 0.05),
    "radius_tire": (0.25, 0.45),
    "c_1x": (5e5, 5e9),
    "c_2x": (5e4, 5e7),
    "C_x": (0.5, 10.0),
    "E_x": (-25.0, 5.0),
}

PER_TRAJ_BOUNDS = {
    "mu": (0.3, 1.6),
}


# --- Fixed physical reference scales for each observation channel ---
# Channels: [yaw_rate, v_x, v_y, a_x, a_y, tire_FL, tire_FR, tire_RL, tire_RR]
# These are typical signal amplitudes, not trajectory-specific stds.
# Using fixed scales avoids pathological blow-up on near-zero channels
# (yaw_rate, v_y are ~0 in straight-line braking — trajectory std ≈ noise floor).
# Tire scales raised to 100 (real range 0–100 rad/s); 30 was too small,
# inflating tire error contributions and drowning out v_x/a_x.
OBS_REF_SCALES = np.array(
    [0.05, 10.0, 0.5, 5.0, 2.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32
)

# --- Parameters to optimize in log scale ---
LOG_SCALE_PARAMS = ["mass", "Inertia_z", "Inertia_tire", "c_1y", "c_2y", "c_1x", "c_2x"]


def log_transform_params(x, param_names):
    return np.array(
        [
            np.log(x[i]) if name in LOG_SCALE_PARAMS else x[i]
            for i, name in enumerate(param_names)
        ]
    )


def exp_transform_params(x, param_names):
    return np.array(
        [
            np.exp(x[i]) if name in LOG_SCALE_PARAMS else x[i]
            for i, name in enumerate(param_names)
        ]
    )


# ============================================================================
# Data preparation
# ============================================================================


def classify_surface(filename: str) -> str:
    """Classify road surface from CSV filename."""
    name_lower = filename.lower()
    if "asphalt" in name_lower:
        return "Asphalt"
    elif "beton" in name_lower:
        return "Concrete"
    elif "basalt" in name_lower:
        return "Basalt"
    return "Unknown"


def prepare_real_data(df: pd.DataFrame, start_idx: int = 50, T_seg: int = 800) -> tuple:
    """
    Extract observations (9-dim) and controls (4-channel dict) from a
    measurement CSV DataFrame.

    Returns
    -------
    y_real : np.ndarray, shape (T_seg, 9)
    controls : dict[str, jnp.ndarray], each shape (T_seg,)
    """
    T_seg = min(T_seg, len(df) - start_idx)

    y_real = np.zeros((T_seg, 9), dtype=np.float32)

    # yaw rate (deg/s → rad/s)
    y_real[:, 0] = (
        df["Rate_Body_Z"].values[start_idx : start_idx + T_seg] * np.pi / 180.0
    )
    # velocities (m/s)
    y_real[:, 1] = df["INS_Vel_Body_X"].values[start_idx : start_idx + T_seg]
    y_real[:, 2] = df["INS_Vel_Body_Y"].values[start_idx : start_idx + T_seg]
    # accelerations (m/s²)
    y_real[:, 3] = df["Acc_Body_X"].values[start_idx : start_idx + T_seg]
    y_real[:, 4] = df["Acc_Body_Y"].values[start_idx : start_idx + T_seg]
    # tire rates (rad/s)
    y_real[:, 5] = df["Tire_Rate_FL"].values[start_idx : start_idx + T_seg]
    y_real[:, 6] = df["Tire_Rate_FR"].values[start_idx : start_idx + T_seg]
    y_real[:, 7] = df["Tire_Rate_RL"].values[start_idx : start_idx + T_seg]
    y_real[:, 8] = df["Tire_Rate_RR"].values[start_idx : start_idx + T_seg]

    # controls  —  steer angle: deg → rad
    controls = {
        "steer_ang": jnp.array(
            np.deg2rad(df["Steer_Angle"].values[start_idx : start_idx + T_seg]),
            dtype=jnp.float32,
        ),
        "engine_torque": jnp.array(
            df["Engine_Torque"].values[start_idx : start_idx + T_seg],
            dtype=jnp.float32,
        ),
        "break_torque": jnp.array(
            df["Break_Pressure"].values[start_idx : start_idx + T_seg],
            dtype=jnp.float32,
        ),
        "gear_transmission": jnp.array(
            df["Gear_Transmission"].values[start_idx : start_idx + T_seg],
            dtype=jnp.float32,
        ),
    }

    return y_real, controls


def load_trajectories(data_dir: Path, start_idx: int, T_seg: int) -> list:
    """Load and prepare all CSV trajectories from data_dir."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        sys.exit(f"ERROR: No CSV files found in {data_dir}")

    prepared = []
    for f in csv_files:
        df = pd.read_csv(f)
        surface = classify_surface(f.stem)
        y_real, controls = prepare_real_data(df, start_idx=start_idx, T_seg=T_seg)
        prepared.append(
            {
                "name": f.stem,
                "surface": surface,
                "y_real": y_real,
                "controls": controls,
                "T": y_real.shape[0],
            }
        )
        print(f"  [{len(prepared)}] {f.stem}  surface={surface}  T={y_real.shape[0]}")

    return prepared


# ============================================================================
# JIT-compiled simulator
# ============================================================================


@jax.jit
def simulate_trajectory_jit(
    p_inf, state0, controls_arr, params_arr, timesteps, radius_tire_val
):
    """
    JIT-compiled forward simulation of the vehicle model.

    Parameters
    ----------
    p_inf : (3,) — [mu, air_resistance, mass]
    state0 : (10,)
    controls_arr : (T, 4)
    params_arr : (15,) — [c_1x,c_2x,c_1y,c_2y,C_x,C_y,E_x,E_y,
                           C_roll1,C_roll2,mass,Inertia_z,Inertia_tire,
                           Inertia_engine,air_resistance]
    timesteps : (T,) — jnp.arange(T)
    radius_tire_val : scalar

    Returns
    -------
    y_seq : (T, 9) — simulated observations
    """
    params = {
        "c_1x": params_arr[0],
        "c_2x": params_arr[1],
        "c_1y": params_arr[2],
        "c_2y": params_arr[3],
        "C_x": params_arr[4],
        "C_y": params_arr[5],
        "E_x": params_arr[6],
        "E_y": params_arr[7],
        "C_roll1": params_arr[8],
        "C_roll2": params_arr[9],
        "mass": params_arr[10],
        "Inertia_z": params_arr[11],
        "Inertia_tire": params_arr[12],
        "Inertia_engine": params_arr[13],
        "air_resistance": params_arr[14],
        "radius_tire": radius_tire_val,
        "dt": 0.01,
    }

    def body_fn(state, t):
        u_t = {
            "steer_ang": controls_arr[t, 0],
            "engine_torque": controls_arr[t, 1],
            "break_torque": controls_arr[t, 2],
            "gear_transmission": controls_arr[t, 3],
        }
        y_t = vehicle_fy(state, u_t, p_inf, **params)
        state_next = vehicle_RK4x(state, u_t, p_inf, **params)
        return state_next, y_t

    _, y_seq = jax.lax.scan(body_fn, state0, timesteps)
    return y_seq


# ============================================================================
# Parameter decoding
# ============================================================================


def decode_params(x, num_trajectories):
    """
    Decode flat optimisation vector → global dict + list of per-traj dicts (linear scale).
    Replaced at runtime by a log-scale variant when --log-scale is set.

    Layout:
        x[0 : NUM_GLOBAL]                             — 16 global params
        x[NUM_GLOBAL : NUM_GLOBAL + num_trajectories]  — per-trajectory mu
    """
    global_dict = {}
    for i, name in enumerate(GLOBAL_PARAMS):
        global_dict[name] = float(x[i])
    traj_params_list = []
    for traj_idx in range(num_trajectories):
        offset = NUM_GLOBAL + traj_idx * NUM_PER_TRAJ
        traj_dict = {}
        for j, name in enumerate(PER_TRAJ_PARAMS):
            traj_dict[name] = float(x[offset + j])
        traj_params_list.append(traj_dict)
    return global_dict, traj_params_list


# ============================================================================
# Objective function
# ============================================================================


def make_objective(prepared_data):
    """
    Build the objective closure over prepared trajectory data.

    Returns a function  f(x) → float  suitable for pyswarm.pso.
    """
    num_traj = len(prepared_data)

    # Pre-build per-trajectory JAX arrays that don't change between evaluations
    traj_cache = []
    for data in prepared_data:
        T = data["T"]
        y0 = data["y_real"][0]
        state0 = jnp.array(
            [0.0, 0.0, 0.0, y0[0], y0[1], y0[2], y0[5], y0[6], y0[7], y0[8]],
            dtype=jnp.float32,
        )
        controls_arr = jnp.stack(
            [
                data["controls"]["steer_ang"],
                data["controls"]["engine_torque"],
                data["controls"]["break_torque"],
                data["controls"]["gear_transmission"],
            ],
            axis=1,
        )
        timesteps = jnp.arange(T)
        traj_cache.append(
            {
                "state0": state0,
                "controls_arr": controls_arr,
                "timesteps": timesteps,
                "y_real": data["y_real"],
                "T": T,
            }
        )

    def objective(x):
        gp, tp_list = decode_params(x, num_traj)

        # Build the params_arr once (same for all trajectories)
        params_arr = jnp.array(
            [
                gp["c_1x"],
                gp["c_2x"],
                gp["c_1y"],
                gp["c_2y"],
                gp["C_x"],
                gp["C_y"],
                gp["E_x"],
                gp["E_y"],
                gp["C_roll1"],
                gp["C_roll2"],
                gp["mass"],
                gp["Inertia_z"],
                gp["Inertia_tire"],
                gp["Inertia_engine"],
                gp["air_resistance"],
            ],
            dtype=jnp.float32,
        )
        radius_tire_val = gp["radius_tire"]

        total_error = 0.0
        for traj_idx in range(num_traj):
            mu = tp_list[traj_idx]["mu"]
            p_inf = jnp.array([mu, gp["air_resistance"], gp["mass"]], dtype=jnp.float32)

            tc = traj_cache[traj_idx]
            try:
                y_sim = simulate_trajectory_jit(
                    p_inf,
                    tc["state0"],
                    tc["controls_arr"],
                    params_arr,
                    tc["timesteps"],
                    radius_tire_val,
                )
                y_sim = np.asarray(y_sim)

                # Smooth penalty — proportional to divergence extent
                nan_mask = np.any(np.isnan(y_sim), axis=1)
                n_valid = int(np.sum(~nan_mask))

                if n_valid == 0:
                    total_error += 1e6  # completely diverged
                    continue

                # Per-channel NRMSE: RMSE per channel, normalise, then average.
                # This gives each channel equal 1/9 weight instead of letting
                # high-dimensional tire channels (4 of 9) dominate pooled MSE.
                y_real_valid = tc["y_real"][~nan_mask]
                y_sim_valid = y_sim[~nan_mask]
                per_ch_rmse = np.sqrt(
                    np.mean((y_sim_valid - y_real_valid) ** 2, axis=0)
                )  # (9,)
                nrmse = float(np.mean(per_ch_rmse / OBS_REF_SCALES))
                divergence_penalty = (1.0 - n_valid / tc["T"]) * 1e4
                total_error += nrmse + divergence_penalty
            except Exception:
                total_error += 1e6  # smoothly penalise crashes too

        return total_error / num_traj

    return objective


# ============================================================================
# Bounds builder
# ============================================================================


def build_bounds(num_trajectories):
    """Return (lb, ub, param_names) arrays for all parameters."""
    lb, ub, names = [], [], []

    # Global
    for name in GLOBAL_PARAMS:
        lo, hi = GLOBAL_BOUNDS[name]
        lb.append(lo)
        ub.append(hi)
        names.append(name)

    # Per-trajectory mu
    for traj_idx in range(num_trajectories):
        for name in PER_TRAJ_PARAMS:
            lo, hi = PER_TRAJ_BOUNDS[name]
            lb.append(lo)
            ub.append(hi)
            names.append(f"{name}_traj{traj_idx + 1}")

    return np.array(lb), np.array(ub), names


# ============================================================================
# Main
# ============================================================================


# ============================================================================
# Advanced PSO engine
# ============================================================================


def advanced_pso(
    objective_fn,
    lb,
    ub,
    swarm_size=50,
    max_iter=500,
    w_start=0.9,
    w_end=0.4,
    c1=1.5,
    c2=1.5,
    topology_switch_frac=0.7,
    stagnation_limit=15,
    reinit_fraction=0.3,
    vmax_fraction=0.5,
    ring_k=2,
    init_positions=None,
    seed=42,
    callback=None,
):
    """
    Advanced PSO with:
      - Internal [0,1] normalisation (all dims comparable)
      - lbest ring topology → gbest switch at *topology_switch_frac*
      - Linear inertia schedule w_start → w_end
      - Velocity clamping
      - Stagnation detection + partial reinit of worst particles

    Parameters
    ----------
    objective_fn : callable
        f(x) → float, where x is in the original (lb, ub) space.
    lb, ub : 1-D arrays
        Bounds in the original space.
    init_positions : (N, D) array in original space, or None.
        If provided, the first len(init_positions) particles are placed here;
        remaining slots (if any) are filled with uniform random samples.
    callback : callable or None
        Called as callback(iteration, gbest_pos_original, gbest_fit, history).

    Returns
    -------
    gbest_pos : 1-D array in original space
    gbest_fit : float
    history : list[float]  – per-iteration best fitness
    """
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    ndim = len(lb)
    span = ub - lb
    span = np.where(span == 0, 1.0, span)  # avoid div-by-zero for fixed params

    switch_iter = int(topology_switch_frac * max_iter)

    # ---- [0,1] normalisation helpers ----
    def to_unit(x):
        return (x - lb) / span

    def from_unit(x01):
        return lb + x01 * span

    def obj(x01):
        return objective_fn(from_unit(np.clip(x01, 0.0, 1.0)))

    # ---- Initialise positions in [0,1]^D ----
    if init_positions is not None:
        n_provided = min(len(init_positions), swarm_size)
        positions = np.clip(
            np.array([to_unit(p) for p in init_positions[:n_provided]]), 0.0, 1.0
        )
        if n_provided < swarm_size:
            extra = rng.uniform(0, 1, (swarm_size - n_provided, ndim))
            positions = np.vstack([positions, extra])
    else:
        positions = rng.uniform(0, 1, (swarm_size, ndim))

    # ---- Velocities (small initial) ----
    vmax = vmax_fraction  # in [0,1] normalised space
    velocities = rng.uniform(-vmax * 0.1, vmax * 0.1, (swarm_size, ndim))

    # ---- Personal bests ----
    pbest_pos = positions.copy()
    pbest_fit = np.full(swarm_size, np.inf)

    print(f"[PSO] Evaluating initial swarm ({swarm_size} particles) …")
    for i in range(swarm_size):
        pbest_fit[i] = obj(positions[i])

    # ---- Global best ----
    gbest_idx = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])
    print(f"[PSO] Initial best = {gbest_fit:.6f}")

    # ---- lbest ring helper ----
    def get_lbest(idx):
        best_f = pbest_fit[idx]
        best_p = pbest_pos[idx].copy()
        for offset in range(-ring_k, ring_k + 1):
            j = (idx + offset) % swarm_size
            if pbest_fit[j] < best_f:
                best_f = pbest_fit[j]
                best_p = pbest_pos[j].copy()
        return best_p

    # ---- Main loop ----
    history = []
    stagnation_count = 0
    prev_gbest_fit = gbest_fit

    for iteration in range(max_iter):
        # Linear inertia decay
        w = w_start - (w_start - w_end) * (iteration / max(max_iter - 1, 1))

        # Topology switch announcement
        use_gbest = iteration >= switch_iter
        if iteration == switch_iter and switch_iter > 0:
            print(
                f"  Iter {iteration + 1:4d} | TOPOLOGY → switching from lbest to gbest"
            )

        for i in range(swarm_size):
            r1 = rng.uniform(0, 1, ndim)
            r2 = rng.uniform(0, 1, ndim)

            cognitive = c1 * r1 * (pbest_pos[i] - positions[i])

            if use_gbest:
                social_target = gbest_pos
            else:
                social_target = get_lbest(i)

            social = c2 * r2 * (social_target - positions[i])

            velocities[i] = w * velocities[i] + cognitive + social
            velocities[i] = np.clip(velocities[i], -vmax, vmax)

            positions[i] = positions[i] + velocities[i]
            positions[i] = np.clip(positions[i], 0.0, 1.0)

            fit = obj(positions[i])

            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = positions[i].copy()

                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest_pos = positions[i].copy()
                    gbest_idx = i

        history.append(gbest_fit)

        # ---- Stagnation detection + partial reinit ----
        if gbest_fit < prev_gbest_fit - 1e-10:
            stagnation_count = 0
            prev_gbest_fit = gbest_fit
        else:
            stagnation_count += 1

        if stagnation_count >= stagnation_limit:
            n_reinit = max(1, int(reinit_fraction * swarm_size))
            worst_indices = np.argsort(pbest_fit)[-n_reinit:]
            reinited = 0
            for idx in worst_indices:
                if idx == gbest_idx:
                    continue  # never reinit the global best particle
                positions[idx] = rng.uniform(0, 1, ndim)
                velocities[idx] = rng.uniform(-vmax * 0.1, vmax * 0.1, ndim)
                # Full reset: also reset pbest so cognitive attraction
                # doesn't pull the particle back to the old basin
                new_fit = obj(positions[idx])
                pbest_pos[idx] = positions[idx].copy()
                pbest_fit[idx] = new_fit
                if new_fit < gbest_fit:
                    gbest_fit = new_fit
                    gbest_pos = positions[idx].copy()
                    gbest_idx = idx
                reinited += 1
            stagnation_count = 0
            print(
                f"  Iter {iteration + 1:4d} | STAGNATION → reinitialised {reinited} particles"
            )

        if callback:
            callback(iteration, from_unit(gbest_pos), gbest_fit, history)

    return from_unit(gbest_pos), gbest_fit, history


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="Advanced PSO optimisation of vehicle model parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="../data/measurements",
        help="Directory containing measurement CSV files",
    )
    p.add_argument("--start-idx", type=int, default=50)
    p.add_argument("--T-seg", type=int, default=800)
    p.add_argument("--swarm-size", type=int, default=50)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--w-start", type=float, default=0.9, help="Initial inertia weight")
    p.add_argument("--w-end", type=float, default=0.4, help="Final inertia weight")
    p.add_argument("--c1", type=float, default=1.5, help="Cognitive coefficient")
    p.add_argument("--c2", type=float, default=1.5, help="Social coefficient")
    p.add_argument(
        "--topology-switch-frac",
        type=float,
        default=0.7,
        help="Fraction of iterations using lbest ring before switching to gbest",
    )
    p.add_argument(
        "--stagnation-limit",
        type=int,
        default=15,
        help="Iterations without improvement before reinitialising worst particles",
    )
    p.add_argument(
        "--reinit-fraction",
        type=float,
        default=0.3,
        help="Fraction of swarm to reinitialise on stagnation",
    )
    p.add_argument(
        "--no-polish",
        action="store_true",
        help="Disable local Powell optimisation after PSO",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--output",
        type=str,
        default="notebooks/experiments/pso_global_optimization_results.json",
        help="Output JSON path (relative to code/)",
    )
    p.add_argument(
        "--log-scale",
        action="store_true",
        help="Optimise selected parameters in log scale (mass, inertia, c_1x, c_2x, etc.)",
    )
    p.add_argument(
        "--warm-start",
        type=str,
        default=None,
        help="Path to a previous PSO results JSON to seed the first particle from",
    )
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()

    print("=" * 70)
    print("Advanced PSO — Global Tire Params, Per-Trajectory mu Only")
    print("=" * 70)
    print(f"  Data dir         : {data_dir}")
    print(f"  T_seg            : {args.T_seg}")
    print(f"  Swarm size       : {args.swarm_size}")
    print(f"  Max iter         : {args.max_iter}")
    print(f"  Inertia          : {args.w_start} → {args.w_end}")
    print(f"  c1, c2           : {args.c1}, {args.c2}")
    print(f"  Topology switch  : lbest → gbest at {args.topology_switch_frac:.0%}")
    print(
        f"  Stagnation reinit: worst {args.reinit_fraction:.0%} after {args.stagnation_limit} iters"
    )
    print(f"  Local polish     : {'off' if args.no_polish else 'Powell'}")
    print(f"  Log scale        : {args.log_scale}")
    print(f"  Warm-start       : {args.warm_start or 'none'}")
    print(f"  Seed             : {args.seed}")
    print(f"  JAX backend      : {jax.default_backend()}")
    print("=" * 70)

    # ---- Load data ----
    print("\n[DATA] Loading trajectories …")
    prepared_data = load_trajectories(
        data_dir, start_idx=args.start_idx, T_seg=args.T_seg
    )
    num_traj = len(prepared_data)
    total_params = NUM_GLOBAL + NUM_PER_TRAJ * num_traj
    print(
        f"\n[PSO] {NUM_GLOBAL} global + {num_traj} × {NUM_PER_TRAJ} per-traj"
        f" = {total_params} params\n"
    )

    # ---- Bounds ----
    lb, ub, param_names = build_bounds(num_traj)

    # Log-scale transform on bounds for selected params
    if args.log_scale:
        for i, name in enumerate(param_names):
            if name in LOG_SCALE_PARAMS:
                lb[i] = np.log(lb[i])
                ub[i] = np.log(ub[i])
        print(f"[LOG-SCALE] Optimising {LOG_SCALE_PARAMS} in log space.")

    # ---- Set decode_params depending on log_scale ----
    global decode_params

    def decode_params_log(x, num_trajectories):
        global_dict = {}
        for i, name in enumerate(GLOBAL_PARAMS):
            val = np.exp(x[i]) if name in LOG_SCALE_PARAMS else x[i]
            global_dict[name] = float(val)
        traj_params_list = []
        for traj_idx in range(num_trajectories):
            offset = NUM_GLOBAL + traj_idx * NUM_PER_TRAJ
            traj_dict = {}
            for j, pname in enumerate(PER_TRAJ_PARAMS):
                traj_dict[pname] = float(x[offset + j])
            traj_params_list.append(traj_dict)
        return global_dict, traj_params_list

    def decode_params_linear(x, num_trajectories):
        global_dict = {}
        for i, name in enumerate(GLOBAL_PARAMS):
            global_dict[name] = float(x[i])
        traj_params_list = []
        for traj_idx in range(num_trajectories):
            offset = NUM_GLOBAL + traj_idx * NUM_PER_TRAJ
            traj_dict = {}
            for j, name in enumerate(PER_TRAJ_PARAMS):
                traj_dict[name] = float(x[offset + j])
            traj_params_list.append(traj_dict)
        return global_dict, traj_params_list

    decode_params = decode_params_log if args.log_scale else decode_params_linear

    # ---- Build objective ----
    objective = make_objective(prepared_data)

    # ---- Warm-up JIT (first call is slow) ----
    print("[JIT] Warm-up call …")
    x_default = np.array([DEFAULTS.get(n.split("_traj")[0], 0.8) for n in param_names])
    if args.log_scale:
        for i, name in enumerate(param_names):
            if name in LOG_SCALE_PARAMS:
                x_default[i] = np.log(x_default[i])
    t0 = datetime.now()
    err0 = objective(x_default)
    print(
        f"[JIT] Warm-up done in {datetime.now() - t0}.  "
        f"Default-param error = {err0:.6f}\n"
    )

    # ---- LHS initialisation with seeded defaults (tip #6) ----
    print("[INIT] Latin Hypercube Sampling + seeded defaults …")
    lhs_sampler = LatinHypercube(d=total_params, seed=args.seed)
    lhs_01 = lhs_sampler.random(n=args.swarm_size)
    init_positions = lb + lhs_01 * (ub - lb)
    init_positions[0] = x_default  # first particle = known-good defaults

    # ---- Warm-start from previous run ----
    if args.warm_start:
        ws_path = Path(args.warm_start)
        if not ws_path.is_absolute():
            ws_path = (PROJECT_ROOT / ws_path).resolve()
        with open(ws_path) as f:
            ws_data = json.load(f)
        ws_x = np.zeros(total_params)
        ws_gp = ws_data["global_params"]
        for i, name in enumerate(GLOBAL_PARAMS):
            val = ws_gp[name]
            if args.log_scale and name in LOG_SCALE_PARAMS:
                val = np.log(val)
            ws_x[i] = val
        for traj_idx, tp in enumerate(ws_data["trajectory_params"][:num_traj]):
            offset = NUM_GLOBAL + traj_idx * NUM_PER_TRAJ
            for j, pname in enumerate(PER_TRAJ_PARAMS):
                ws_x[offset + j] = tp["params"][pname]
        ws_x = np.clip(ws_x, lb, ub)
        init_positions[0] = ws_x  # override default seed with previous best
        print(f"[WARM-START] Loaded previous best from {ws_path}")
        print(f"[WARM-START] Previous error: {ws_data.get('final_error', 'N/A')}")

    n_seeded = "warm-start" if args.warm_start else "default seed"
    print(
        f"[INIT] {args.swarm_size} particles "
        f"({args.swarm_size - 1} LHS + 1 {n_seeded})"
    )

    # ---- Iteration callback for logging ----
    start_time = [datetime.now()]

    def iteration_callback(iteration, gbest_pos, gbest_fit, history):
        if (iteration + 1) % 10 == 0:
            elapsed = datetime.now() - start_time[0]
            print(
                f"  Iter {iteration + 1:4d} | Best = {gbest_fit:.6f} "
                f"| Elapsed = {elapsed}"
            )

    # ---- Run advanced PSO ----
    print(
        f"\n[PSO] Starting optimisation "
        f"({args.max_iter} iters, swarm={args.swarm_size}) …\n"
    )
    start_time[0] = datetime.now()

    best_pos, best_fit, convergence_history = advanced_pso(
        objective,
        lb,
        ub,
        swarm_size=args.swarm_size,
        max_iter=args.max_iter,
        w_start=args.w_start,
        w_end=args.w_end,
        c1=args.c1,
        c2=args.c2,
        topology_switch_frac=args.topology_switch_frac,
        stagnation_limit=args.stagnation_limit,
        reinit_fraction=args.reinit_fraction,
        init_positions=init_positions,
        seed=args.seed,
        callback=iteration_callback,
    )

    elapsed_pso = datetime.now() - start_time[0]
    total_evals = args.swarm_size * (1 + args.max_iter)
    print(f"\n[PSO] Finished in {elapsed_pso}")
    print(f"[PSO] Best fitness (mean weighted MSE): {best_fit:.6f}")
    print(f"[PSO] Approx evaluations: {total_evals}")

    # ---- Local polish with Powell (tip #7) ----
    if not args.no_polish:
        print("\n[POLISH] Running Powell local optimisation around PSO best …")
        t_polish = datetime.now()

        def bounded_objective(x):
            return objective(np.clip(x, lb, ub))

        polish_result = minimize(
            bounded_objective,
            best_pos,
            method="Powell",
            options={"maxiter": 5000, "ftol": 1e-10},
        )
        polish_pos = np.clip(polish_result.x, lb, ub)
        polish_fit = objective(polish_pos)

        if polish_fit < best_fit:
            improvement = best_fit - polish_fit
            print(
                f"[POLISH] Improved: {best_fit:.6f} → {polish_fit:.6f} "
                f"(Δ = {improvement:.6f})"
            )
            best_pos = polish_pos
            best_fit = polish_fit
        else:
            print(f"[POLISH] No improvement ({polish_fit:.6f} >= {best_fit:.6f})")

        elapsed_polish = datetime.now() - t_polish
        print(f"[POLISH] Time: {elapsed_polish}")
        total_evals += polish_result.nfev + 1

    elapsed_total = datetime.now() - start_time[0]

    # ---- Decode results ----
    global_opt, traj_opt_list = decode_params(best_pos, num_traj)

    print("\n[RESULTS] Optimised global parameters:")
    for k, v in global_opt.items():
        default = DEFAULTS.get(k, "?")
        print(f"  {k:20s}: {v:12.6f}  (default: {default})")

    print("\n[RESULTS] Per-trajectory mu:")
    for i, (data, tp) in enumerate(zip(prepared_data, traj_opt_list)):
        print(
            f"  Traj {i+1} [{data['surface']:10s}] "
            f"{data['name'][:50]:50s}  mu = {tp['mu']:.6f}"
        )

    # ---- Per-trajectory error breakdown ----
    print("\n[RESULTS] Per-trajectory error breakdown:")
    for i, data in enumerate(prepared_data):
        mu = traj_opt_list[i]["mu"]
        p_inf = jnp.array(
            [mu, global_opt["air_resistance"], global_opt["mass"]],
            dtype=jnp.float32,
        )
        params_arr = jnp.array(
            [
                global_opt["c_1x"],
                global_opt["c_2x"],
                global_opt["c_1y"],
                global_opt["c_2y"],
                global_opt["C_x"],
                global_opt["C_y"],
                global_opt["E_x"],
                global_opt["E_y"],
                global_opt["C_roll1"],
                global_opt["C_roll2"],
                global_opt["mass"],
                global_opt["Inertia_z"],
                global_opt["Inertia_tire"],
                global_opt["Inertia_engine"],
                global_opt["air_resistance"],
            ],
            dtype=jnp.float32,
        )
        y0 = data["y_real"][0]
        state0 = jnp.array(
            [0.0, 0.0, 0.0, y0[0], y0[1], y0[2], y0[5], y0[6], y0[7], y0[8]],
            dtype=jnp.float32,
        )
        controls_arr = jnp.stack(
            [
                data["controls"]["steer_ang"],
                data["controls"]["engine_torque"],
                data["controls"]["break_torque"],
                data["controls"]["gear_transmission"],
            ],
            axis=1,
        )
        timesteps = jnp.arange(data["T"])
        y_sim = np.asarray(
            simulate_trajectory_jit(
                p_inf,
                state0,
                controls_arr,
                params_arr,
                timesteps,
                global_opt["radius_tire"],
            )
        )
        per_ch_rmse = np.sqrt(np.mean((y_sim - data["y_real"]) ** 2, axis=0))
        nrmse = float(np.mean(per_ch_rmse / OBS_REF_SCALES))
        print(f"  Traj {i+1} [{data['surface']:10s}]: NRMSE = {nrmse:.6f}")

    # ---- Save results ----
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "method": (
            "Advanced PSO — lbest→gbest, inertia schedule, "
            "LHS init, stagnation reinit, Powell polish"
        ),
        "log_scale": args.log_scale,
        "timestamp": datetime.now().isoformat(),
        "final_error": float(best_fit),
        "approx_function_evaluations": total_evals,
        "swarm_size": args.swarm_size,
        "max_iterations": args.max_iter,
        "w_schedule": f"{args.w_start} → {args.w_end}",
        "c1": args.c1,
        "c2": args.c2,
        "topology_switch_frac": args.topology_switch_frac,
        "stagnation_limit": args.stagnation_limit,
        "reinit_fraction": args.reinit_fraction,
        "local_polish": not args.no_polish,
        "seed": args.seed,
        "warm_start": args.warm_start,
        "T_seg": args.T_seg,
        "start_idx": args.start_idx,
        "num_trajectories": num_traj,
        "total_optimised_params": total_params,
        "elapsed_pso_seconds": elapsed_pso.total_seconds(),
        "elapsed_total_seconds": elapsed_total.total_seconds(),
        "jax_backend": str(jax.default_backend()),
        "global_params": {k: float(v) for k, v in global_opt.items()},
        "trajectory_params": [
            {
                "trajectory": data["name"],
                "surface": data["surface"],
                "params": {k: float(v) for k, v in tp.items()},
            }
            for data, tp in zip(prepared_data, traj_opt_list)
        ],
        "convergence_history": [float(e) for e in convergence_history],
        "defaults_used": {k: float(v) for k, v in DEFAULTS.items()},
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] Results written to {output_path}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print(
        f"DONE — Best error: {best_fit:.6f}  |  "
        f"Evals: ~{total_evals}  |  Time: {elapsed_total}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
