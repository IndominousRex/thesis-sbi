# sbi_vehicle/metrics.py

import torch
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple


def parameter_rmse_distribution(theta_true: torch.Tensor, theta_samples: torch.Tensor):
    """
    Parameter RMSE using the full posterior distribution.
    """
    assert theta_true.ndim == 2
    assert theta_samples.ndim == 3
    K, T, D = theta_samples.shape
    assert theta_true.shape == (T, D)

    theta_true_expanded = theta_true.unsqueeze(1)  # (N, 1, d)
    sq_err = (theta_samples - theta_true_expanded) ** 2  # (N, K, d)

    mse_per_case = sq_err.mean(dim=(1, 2))  # (N,)
    rmse_per_case = torch.sqrt(mse_per_case)
    rmse_overall = rmse_per_case.mean()

    mse_per_dim = sq_err.mean(dim=(0, 1))  # (d,)
    rmse_per_dim = torch.sqrt(mse_per_dim)

    return rmse_per_case, rmse_overall, rmse_per_dim


def posterior_spread_vs_error(
    theta_true: torch.Tensor, theta_samples: torch.Tensor, eps: float = 1e-8
):
    """
    Compare posterior spread (std) vs error of posterior mean.
    """
    assert theta_true.ndim == 2
    assert theta_samples.ndim == 3
    N, K, d = theta_samples.shape
    assert theta_true.shape == (N, d)

    mean_post = theta_samples.mean(dim=1)  # (N, d)
    std_post = theta_samples.std(dim=1, unbiased=True)  # (N, d)
    abs_error = (mean_post - theta_true).abs()  # (N, d)
    z_scores = (mean_post - theta_true) / (std_post + eps)

    return mean_post, std_post, abs_error, z_scores


def w2_sequence_vs_real(
    y_real: torch.Tensor,
    y_samples: torch.Tensor,
    normalize: bool = False,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate W2 distance between a posterior predictive distribution
    over trajectories and a single real trajectory (point mass).
    """
    assert y_real.ndim == 2
    assert y_samples.ndim == 3
    K, T, D = y_samples.shape
    assert y_real.shape == (T, D)

    y_real_exp = y_real.unsqueeze(0)  # (1, T, D)

    if normalize:
        combined = torch.cat([y_real_exp, y_samples], dim=0)
        mean = combined.mean(dim=(0, 1), keepdim=True)
        std = combined.std(dim=(0, 1), keepdim=True)
        y_real_norm = (y_real_exp - mean) / (std + eps)
        y_samples_norm = (y_samples - mean) / (std + eps)
    else:
        y_real_norm = y_real_exp
        y_samples_norm = y_samples

    diffs = y_samples_norm - y_real_norm
    sq_norm_per_sample = (diffs**2).sum(dim=(1, 2))  # (K,)
    w2_sq = sq_norm_per_sample.mean()
    w2 = torch.sqrt(w2_sq)
    w2_per_sample = torch.sqrt(sq_norm_per_sample)

    return w2, w2_per_sample


def sliced_wasserstein_distance(rng, samples_true, samples_model, num_projections=1000):
    """
    JAX implementation of sliced Wasserstein distance between two sets of samples.

    samples_true:  (N, d) jnp array
    samples_model: (N, d) jnp array
    """
    projections = jax.random.normal(rng, (num_projections, samples_true.shape[-1]))
    projections = projections / jnp.linalg.norm(projections, axis=-1, keepdims=True)
    projections = jnp.transpose(projections)  # (d, num_projections)

    samples_true_proj = jnp.dot(samples_true, projections)
    samples_model_proj = jnp.dot(samples_model, projections)

    samples_true_proj = jnp.sort(samples_true_proj, axis=0)
    samples_model_proj = jnp.sort(samples_model_proj, axis=0)

    swd = jnp.mean(jnp.abs(samples_true_proj - samples_model_proj))
    return swd


def sliced_wasserstein_prior_vs_dap(
    prior, dap_samples, num_projections=1000, seed=0
) -> float:
    """
    Compare data-averaged posterior (DAP) samples from SBC to the prior.
    dap_samples: torch.Tensor (N, d)
    """
    N, d = dap_samples.shape
    prior_samples = prior.sample((N,))  # (N, d)

    st = jnp.asarray(prior_samples.detach().cpu().numpy())
    sm = jnp.asarray(dap_samples.detach().cpu().numpy())

    rng = jax.random.PRNGKey(seed)
    swd_val = sliced_wasserstein_distance(rng, st, sm, num_projections=num_projections)
    return float(swd_val)


def real_data_trajectory_metrics(
    y_real_np: np.ndarray,
    y_ppc_np: np.ndarray,
    *,
    normalize_w2: bool = True,
) -> dict:
    """
    Convenience wrapper for computing real-data metrics from numpy arrays.

    Args:
        y_real_np: (T, D) numpy array, real trajectory.
        y_ppc_np:  (K, T, D) numpy array, posterior predictive trajectories.
        normalize_w2: if True, use normalization in W2 to make it scale-free.

    Returns:
        dict with:
          - rmse_overall: scalar
          - rmse_per_dim: (D,) list
          - w2:            scalar
          - w2_mean_per_sample: scalar
    """
    assert y_real_np.ndim == 2
    assert y_ppc_np.ndim == 3
    K, T, D = y_ppc_np.shape
    assert y_real_np.shape == (T, D)

    # Convert to torch
    y_real_t = torch.from_numpy(y_real_np.astype(np.float32))
    y_ppc_t = torch.from_numpy(y_ppc_np.astype(np.float32))

    # 1) RMSE metrics
    _, rmse_overall, rmse_per_dim_t = parameter_rmse_distribution(y_real_t, y_ppc_t)
    rmse_per_dim = rmse_per_dim_t.cpu().numpy().tolist()

    # 2) W2 metrics
    w2, w2_per_sample = w2_sequence_vs_real(
        y_real_t,
        y_ppc_t,
        normalize=normalize_w2,
    )
    w2_val = float(w2)
    w2_mean_per_sample = float(w2_per_sample.mean())

    return {
        "rmse_overall": float(rmse_overall),
        "rmse_per_dim": rmse_per_dim,
        "w2": w2_val,
        "w2_mean_per_sample": w2_mean_per_sample,
    }
