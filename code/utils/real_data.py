from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import torch

from configs.config import ExperimentConfig
from simulation.simulation import (
    controls_to_array,
    rollout_with_states,
    expand_theta_to_full,
)


def controls_from_array_np(ctrl_array: np.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Convert a (T, 4) control array back into the dict layout the simulator expects.
    Order matches simulation.controls_to_array: [steer, engine, brake, gear].
    """
    if ctrl_array.shape[1] != 4:
        raise ValueError(
            f"Expected control array with 4 columns, got {ctrl_array.shape}"
        )
    return {
        "steer_ang": jnp.asarray(ctrl_array[:, 0]),
        "engine_torque": jnp.asarray(ctrl_array[:, 1]),
        "break_torque": jnp.asarray(ctrl_array[:, 2]),
        "gear_transmission": jnp.asarray(ctrl_array[:, 3]),
    }


# Observation labels for real data
OBS_LABELS = [
    "yaw_rate",
    "v_body_x",
    "v_body_y",
    "a_body_x",
    "a_body_y",
    "tire_rate_fl",
    "tire_rate_fr",
    "tire_rate_rl",
    "tire_rate_rr",
]


def prep_x_obs_from_df(
    df: pd.DataFrame,
    start_idx: int,
    T: int,
    *,
    rate_body_z_in_deg_s: bool = True,
    tire_rates_in_rpm: bool = False,
    vel_body_in_kmh: bool = False,
) -> torch.Tensor:
    """
    Build observation vector y_t from the CSV in the same format
    as the simulator output:

      [ yaw_rate, v_x, v_y, a_x, a_y, ω_FL, ω_FR, ω_RL, ω_RR ]

    All returned in SI units (rad/s, m/s, m/s², rad/s).

    Args:
        df:   Real CSV as a pandas DataFrame.
        start_idx: index of first row to use.
        T:    number of time steps.
        rate_body_z_in_deg_s: if True, convert deg/s -> rad/s.
        tire_rates_in_rpm:    if True, convert rpm -> rad/s.
        vel_body_in_kmh:      if True, convert km/h -> m/s.

    Returns:
        x_obs: torch.Tensor of shape (T, 9), dtype float32.
    """
    i0, i1 = start_idx, start_idx + T

    # --- 1) yaw_rate (rad/s) ---
    yaw_rate = df.iloc[i0:i1]["Rate_Body_Z"].to_numpy(dtype=np.float32)
    if rate_body_z_in_deg_s:
        yaw_rate = yaw_rate * (np.pi / 180.0)  # deg/s -> rad/s

    # --- 2) body-frame velocities (m/s) ---
    v_body_x = df.iloc[i0:i1]["INS_Vel_Body_X"].to_numpy(dtype=np.float32)
    v_body_y = df.iloc[i0:i1]["INS_Vel_Body_Y"].to_numpy(dtype=np.float32)
    if vel_body_in_kmh:
        v_body_x *= 1000.0 / 3600.0
        v_body_y *= 1000.0 / 3600.0

    # --- 3) body-frame accelerations (m/s²) ---
    a_body_x = df.iloc[i0:i1]["Acc_Body_X"].to_numpy(dtype=np.float32)
    a_body_y = df.iloc[i0:i1]["Acc_Body_Y"].to_numpy(dtype=np.float32)

    # --- 4) tire angular rates (rad/s) ---
    def get_tire(col: str):
        arr = df.iloc[i0:i1][col].to_numpy(dtype=np.float32)
        if tire_rates_in_rpm:
            arr = arr * (2.0 * np.pi / 60.0)  # rpm -> rad/s
        return arr

    tire_fl = get_tire("Tire_Rate_FL")
    tire_fr = get_tire("Tire_Rate_FR")
    tire_rl = get_tire("Tire_Rate_RL")
    tire_rr = get_tire("Tire_Rate_RR")

    x_obs = np.stack(
        [
            yaw_rate,
            v_body_x,
            v_body_y,
            a_body_x,
            a_body_y,
            tire_fl,
            tire_fr,
            tire_rl,
            tire_rr,
        ],
        axis=1,
    ).astype(np.float32)

    return torch.from_numpy(x_obs)


def make_controls_from_df(
    df: pd.DataFrame,
    start_idx: int,
    T: int,
    *,
    steer_in_deg: bool = True,
) -> Dict[str, jnp.ndarray]:
    """
    Extract control signals from the CSV:

        steer_ang [rad], engine_torque, break_torque, gear_transmission

    Returns:
        dict of jnp arrays, each of length T.
    """
    i0, i1 = start_idx, start_idx + T

    steer = df.iloc[i0:i1]["Steer_Angle"].to_numpy(np.float32)
    if steer_in_deg:
        steer = np.deg2rad(steer)

    eng_t = df.iloc[i0:i1]["Engine_Torque"].to_numpy(np.float32)
    brk_t = df.iloc[i0:i1]["Break_Pressure"].to_numpy(np.float32)
    gear = df.iloc[i0:i1]["Gear_Transmission"].to_numpy(np.float32)

    return {
        "steer_ang": jnp.asarray(steer),
        "engine_torque": jnp.asarray(eng_t),
        "break_torque": jnp.asarray(brk_t),
        "gear_transmission": jnp.asarray(gear),
    }


def initial_state_from_obs(y0: np.ndarray) -> jnp.ndarray:
    """
    Build an initial simulator state from the first observation sample.

    State layout: [geo_x, geo_y, yaw, dyaw, v_x, v_y, tire_fl, tire_fr, tire_rl, tire_rr].
    """
    if y0.shape[0] < 9:
        raise ValueError(f"Expected at least 9 observation dims, got {y0.shape}")

    geo_pos = (0.0, 0.0)
    yaw = 0.0  # unknown absolute yaw; keep zero to avoid drift
    dyaw = float(y0[0])
    v_x, v_y = float(y0[1]), float(y0[2])
    tire_fl, tire_fr, tire_rl, tire_rr = [float(val) for val in y0[5:9]]

    return jnp.asarray(
        [*geo_pos, yaw, dyaw, v_x, v_y, tire_fl, tire_fr, tire_rl, tire_rr],
        dtype=jnp.float32,
    )


def pick_start_idx_len_safe(
    df: pd.DataFrame,
    T_raw: int,
    *,
    prefer_low_brake: bool = True,
    thresh: float = 5.0,
    max_viol_frac: float = 0.01,
) -> int:
    """
    Choose a valid start index in [0, N - T_raw].

    If prefer_low_brake=True, pick the window with the most 'low brake'
    samples (Break_Pressure < thresh) as long as the fraction of
    violations is <= max_viol_frac. Otherwise fall back to the last window.

    Raises a clear error if df has fewer than T_raw rows.
    """
    N = len(df)
    if N < T_raw:
        raise AssertionError(
            f"CSV has only {N} rows, but {T_raw} are required for this trained model."
        )

    if prefer_low_brake:
        brk = df["Break_Pressure"].to_numpy(np.float32)
        ok = (brk < thresh).astype(np.int32)
        win = np.convolve(ok, np.ones(T_raw, dtype=np.int32), mode="valid")
        best_idx = int(np.argmax(win))
        violations = int(T_raw - win[best_idx])
        if violations <= int(max_viol_frac * T_raw):
            return best_idx

    return N - T_raw


def simulate_y_batch_for_thetas(
    theta_batch_np: np.ndarray,
    controls: Dict[str, jnp.ndarray],
    state_dim: int,
    cfg: ExperimentConfig,
    state0: Optional[jnp.ndarray] = None,
) -> np.ndarray:
    """
    Simulate observation trajectories y_t for a batch of theta using the JAX
    vehicle model and the standardized vehicle_fy measurement.

    Args:
        theta_batch_np: (K, d_active) numpy array of theta samples.
        controls:       dict of controls at *model rate* (length T_event), in the
                        same format as training:
                        { "steer_ang", "engine_torque", "break_torque",
                          "gear_transmission" }.
        state_dim:      dimension of the state vector (10 in this model).
        cfg:            ExperimentConfig with active param selection.
        state0:         optional initial state; if None, uses zeros.

    Returns:
        y_batch: (K, T_event, obs_dim) numpy array of observations only.
                 obs_dim must match vehicle_fy output (9).
    """
    # Expand from active parameters to full parameter vector
    theta_full = expand_theta_to_full(theta_batch_np, cfg)  # (K, d_full)
    p_batch = jnp.asarray(theta_full, dtype=jnp.float32)  # (K, d_full)

    # Prepare initial state
    if state0 is None:
        state0_jnp = jnp.zeros(state_dim, jnp.float32)
    else:
        state0_arr = np.asarray(state0, dtype=np.float32)
        if state0_arr.shape[0] != state_dim:
            raise ValueError(
                f"state0 has shape {state0_arr.shape}, expected ({state_dim},)"
            )
        state0_jnp = jnp.asarray(state0_arr, dtype=jnp.float32)

    # Single-trajectory rollout: rollout_with_states returns (state_seq, y_seq)
    def one(p):
        return rollout_with_states(p, controls, state0=state0_jnp)[1]  # y_seq

    # Vectorize over theta and JIT-compile
    y_batch = jax.jit(jax.vmap(one, in_axes=(0,)))(p_batch)  # (K, T_event, obs_dim)
    return np.asarray(y_batch, np.float32)


def build_real_window_from_csv(
    df_real: pd.DataFrame,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    start_idx: Optional[int] = None,
    prefer_low_brake: bool = True,
    brake_thresh: float = 5.0,
    max_viol_frac: float = 0.01,
    rate_body_z_in_deg_s: bool = True,
    tire_rates_in_rpm: bool = False,
    vel_body_in_kmh: bool = False,
) -> Tuple[torch.Tensor, Dict[str, jnp.ndarray], int]:
    """
    Construct x_obs_full and controls_real from the real CSV, in the
    *same layout and time resolution* as the training data.

    Training pipeline:
      - simulates length T_seg at native rate,
      - computes y = vehicle_fy(state_t, u_t, theta),
      - concatenates [y || controls] -> (T_seg, obs_dim+4),

    This function mirrors that:
      - builds y_real_raw and controls_real at raw rate,
      - concatenates [y_raw || controls_raw],
      - returns (1, T_event, D_in) tensor plus the *raw-rate* controls dict.
    """
    # 0) Raw length to extract (before decimation)
    T_raw = cfg.T_seg
    N = len(df_real)

    if start_idx is None:
        start_idx = pick_start_idx_len_safe(
            df_real,
            T_raw,
            prefer_low_brake=prefer_low_brake,
            thresh=brake_thresh,
            max_viol_frac=max_viol_frac,
        )

    print(f"[real window] start_idx={start_idx}, rows={T_raw}, total_rows={N}")

    # 1) Observations at raw rate (T_raw, obs_dim)
    #    Layout matches vehicle_fy: [yaw_rate, v_body_x, v_body_y, a_body_x, a_body_y,
    #                                tire_fl, tire_fr, tire_rl, tire_rr]
    x_obs_raw = prep_x_obs_from_df(
        df_real,
        start_idx=start_idx,
        T=T_raw,
        rate_body_z_in_deg_s=rate_body_z_in_deg_s,
        tire_rates_in_rpm=tire_rates_in_rpm,
        vel_body_in_kmh=vel_body_in_kmh,
    )  # (T_raw, obs_dim), numpy float32

    # 2) Controls at raw rate, in same format as training simulator
    controls_real = make_controls_from_df(
        df_real,
        start_idx=start_idx,
        T=T_raw,
        steer_in_deg=True,
    )
    c_real = np.asarray(
        controls_to_array(controls_real), dtype=np.float32
    )  # (T_raw, 4)

    # 3) Concatenate in the same order as training: [obs || controls]
    x_concat = np.concatenate([x_obs_raw, c_real], axis=-1)  # (T_raw, D_in

    # 4) Convert to torch tensor for the model: (1, T_event, D_in)
    x_obs_full = torch.from_numpy(x_concat.astype(np.float32)).unsqueeze(0).to(device)

    return x_obs_full, controls_real, start_idx


def build_simulated_window_for_eval(
    cfg: ExperimentConfig,
    prior,
    simulator,
    device: torch.device,
    *,
    control_offset_batches: int = 1000,
) -> Tuple[torch.Tensor, Dict[str, jnp.ndarray]]:
    """
    Generate a simulated window (not from training data) to use as a PPC sanity check.

    We offset the simulator's internal batch index to avoid reusing control recipes seen in
    training and sample a new theta from the prior.
    """
    with torch.no_grad():
        # Nudge internal counter so control recipes differ from training set
        if hasattr(simulator, "_batch_idx"):
            simulator._batch_idx = getattr(simulator, "_batch_idx", 0) + max(
                control_offset_batches, getattr(cfg, "num_simulations", 0)
            )

        theta_sim = prior.sample((1,)).to(device)
        x_sim_full = simulator(theta_sim).detach()  # (1, T, D_in)

    ctrl_array = x_sim_full[0, :, cfg.obs_dim :].cpu().numpy()
    controls_sim = controls_from_array_np(ctrl_array)

    return x_sim_full.to(device), controls_sim


def posterior_predictive_from_real(
    posterior,
    x_obs_full: torch.Tensor,
    controls_real: Dict[str, jnp.ndarray],
    cfg: ExperimentConfig,
    K_ppc: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Posterior predictive simulation for a real (or simulated) window.

    Args:
        posterior:     trained sbi posterior.
        x_obs_full:    (1, T_event, D_in) torch tensor on 'device'.
                       Layout must be [obs_dim || 4 control channels].
        controls_real: dict of controls at raw rate (length T_raw), in the same
                       format as used by the training simulator
                       (steer_ang, engine_torque, break_torque, gear_transmission).
        cfg:           ExperimentConfig.
        K_ppc:         number of posterior predictive trajectories.

    Returns:
        y_real : (T_event, obs_dim) numpy array (real observations).
        y_ppc  : (K_ppc, T_event, obs_dim) numpy array (simulated).
    """
    # 1) Real observations at model rate: first obs_dim dims only
    #    This matches vehicle_fy output ordering:
    #    [yaw_rate, v_body_x, v_body_y, a_body_x, a_body_y, tire_fl, tire_fr, tire_rl, tire_rr]
    y_real = (
        x_obs_full[0, :, : cfg.obs_dim].detach().cpu().numpy().astype(np.float32)
    )  # (T_event, obs_dim)

    T_event = y_real.shape[0]

    # 2) Match control rate to model rate
    #    build_real_window_from_csv gives controls at raw rate; if training
    #    used decimation, we apply the same here.
    ctrls_model = controls_real

    # Basic sanity: steer_ang length must match T_event (or be safely truncatable)
    L_ctrl = int(ctrls_model["steer_ang"].shape[0])
    if L_ctrl != T_event:
        # Truncate conservatively to min length, to avoid shape errors
        L_min = min(L_ctrl, T_event)
        ctrls_model = {k: v[:L_min] for k, v in ctrls_model.items()}
        y_real = y_real[:L_min]
        T_event = L_min

    # 3) Use the first real observation to seed the simulator state.
    #    This maps y0 → full state [geo_x, geo_y, yaw, dyaw, v_x, v_y, tire_fl, tire_fr, tire_rl, tire_rr].
    state0_real = initial_state_from_obs(y_real[0])

    # 4) Sample parameters from p(theta | x_real)
    with torch.no_grad():
        samples = posterior.sample((K_ppc,), x=x_obs_full)
        # samples can be (K, d) or (K, C, d) if using multiple chains
        samples = samples.cpu()
        if samples.ndim == 3:
            # assume shape (K, num_chains, d) -> take chain 0
            samples = samples[:, 0, :]
        elif samples.ndim != 2:
            raise ValueError(f"Unexpected posterior sample shape {samples.shape}")

    thetas = samples.numpy().astype(np.float32)  # (K_ppc, d_active)

    # 5) Simulate y for each theta using the JAX vehicle model
    #    This uses rollout_with_states + vehicle_fy under the hood.
    y_ppc = simulate_y_batch_for_thetas(
        theta_batch_np=thetas,
        controls=ctrls_model,
        state_dim=cfg.state_dim,
        cfg=cfg,
        state0=state0_real,
    )  # (K_ppc, T_event, obs_dim)

    return y_real, y_ppc
