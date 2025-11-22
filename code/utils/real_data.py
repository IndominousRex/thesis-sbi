from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import torch

from ..configs.config import ExperimentConfig
from ..simulation.simulation import (
    controls_to_array,
    rollout_with_states,
    expand_theta_to_full,
)


# Observation labels for real data
OBS_LABELS = [
    "yaw_rate [rad/s]",
    "v_body_x [m/s]",
    "v_body_y [m/s]",
    "a_body_x [m/s²]",
    "a_body_y [m/s²]",
    "tire_FL [rad/s]",
    "tire_FR [rad/s]",
    "tire_RL [rad/s]",
    "tire_RR [rad/s]",
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


def decimate_controls_dict(
    ctrls: Dict[str, jnp.ndarray],
    factor: int,
) -> Dict[str, jnp.ndarray]:
    """
    Decimate a control dictionary: keep every 'factor'-th sample along time,
    truncating to a multiple of 'factor'.
    """
    if factor <= 1:
        return ctrls

    L = int(ctrls["steer_ang"].shape[0]) // factor * factor
    return {
        k: jnp.asarray(np.asarray(v)[:L:factor], dtype=jnp.float32)
        for k, v in ctrls.items()
    }


def simulate_y_batch_for_thetas(
    theta_batch_np: np.ndarray,
    controls_dec: Dict[str, jnp.ndarray],
    state_dim: int,
    cfg: ExperimentConfig,
) -> np.ndarray:
    """
    Simulate the observation trajectories y_t for a batch of theta using the
    JAX vehicle model, expanding a partial theta to the full (mu, cd, m).

    Args:
        theta_batch_np: (K, d_active) numpy array of theta samples.
        controls_dec:   dict of decimated controls (length T_event).
        state_dim:      dimension of the state vector (10 in your model).
        cfg:            experiment config with active param selection.

    Returns:
        y_batch: (K, T_event, OBS_D) numpy array (obs only, no controls).
    """
    import jax.numpy as jnp_local  # just alias

    theta_full = expand_theta_to_full(theta_batch_np, cfg)
    p_batch = jnp_local.asarray(theta_full, dtype=jnp.float32)
    state0 = jnp_local.zeros(state_dim, jnp.float32)

    def one(p):
        # rollout_with_states returns (state_seq, y_seq); we take y_seq
        return rollout_with_states(p, controls_dec, state0=state0)[1]

    y_batch = jax.jit(jax.vmap(one, in_axes=(0,)))(p_batch)  # (K, T, OBS_D)
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
    *same layout and length* as the training data.

    Training pipeline did:
      - simulate length T_seg at raw rate
      - observations y (obs_dim) + controls (4) → concat → (T_seg, D_in)
      - decimate by cfg.decimate → (T_event, D_in)

    Here we replicate the same steps.
    """
    # Raw length before decimation: should match training T_seg
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

    # 1) Observations at raw rate
    x_obs_raw = prep_x_obs_from_df(
        df_real,
        start_idx=start_idx,
        T=T_raw,
        rate_body_z_in_deg_s=rate_body_z_in_deg_s,
        tire_rates_in_rpm=tire_rates_in_rpm,
        vel_body_in_kmh=vel_body_in_kmh,
    )  # (T_raw, obs_dim)

    # 2) Controls at raw rate
    controls_real = make_controls_from_df(df_real, start_idx, T_raw, steer_in_deg=True)
    c_real = np.asarray(controls_to_array(controls_real))  # (T_raw, 4)

    # 3) Concat in same order as training: [obs || controls]
    x_concat = np.concatenate([x_obs_raw.numpy(), c_real], axis=-1)  # (T_raw, D_in)

    # 4) Apply the same decimation as used in training
    if cfg.decimate > 1:
        L = (x_concat.shape[0] // cfg.decimate) * cfg.decimate
        x_concat = x_concat[: L : cfg.decimate]

    x_obs_full = torch.from_numpy(x_concat.astype(np.float32)).unsqueeze(0).to(device)
    # shape: (1, T_event, D_in)

    return x_obs_full, controls_real, start_idx


def posterior_predictive_from_real(
    posterior,
    x_obs_full: torch.Tensor,
    controls_real: Dict[str, jnp.ndarray],
    cfg: ExperimentConfig,
    K_ppc: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Posterior predictive simulation for a real CSV window.

    Args:
        posterior:     trained sbi posterior.
        x_obs_full:    (1, T_event, D_in) torch tensor on 'device'.
        controls_real: dict of controls at raw rate (length T_raw).
        cfg:           ExperimentConfig.
        K_ppc:         number of posterior predictive trajectories.

    Returns:
        y_real : (T_event, obs_dim) numpy array (real observations).
        y_ppc  : (K_ppc, T_event, obs_dim) numpy array (simulated).
    """
    # Real observations at model rate: first obs_dim dims only
    y_real = x_obs_full[0, :, : cfg.obs_dim].detach().cpu().numpy().astype(np.float32)

    # Match control rate to model rate
    ctrls_dec = decimate_controls_dict(controls_real, cfg.decimate)

    # Sample parameters from p(theta | x_real)
    with torch.no_grad():
        thetas = (
            posterior.sample((K_ppc,), x=x_obs_full).cpu().numpy().astype(np.float32)
        )  # (K_ppc, d_active)

    # Simulate y for each theta using JAX
    y_ppc = simulate_y_batch_for_thetas(
        thetas, ctrls_dec, state_dim=cfg.state_dim, cfg=cfg
    )

    return y_real, y_ppc
