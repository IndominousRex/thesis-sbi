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
    N, K, d = theta_samples.shape
    assert theta_true.shape == (N, d)

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


def rmse_posterior_predictive(y_real: torch.Tensor, y_samples: torch.Tensor):
    """
    RMSE between posterior predictive trajectories and real trajectory.
    """
    assert y_real.ndim == 2
    assert y_samples.ndim == 3
    K, T, D = y_samples.shape
    assert y_real.shape == (T, D)

    y_real_expanded = y_real.unsqueeze(0)  # (1, T, D)

    sq_err = (y_samples - y_real_expanded) ** 2
    mse_per_sample = sq_err.mean(dim=(1, 2))  # (K,)
    rmse_per_sample = torch.sqrt(mse_per_sample)
    rmse_overall = rmse_per_sample.mean()

    mse_per_dim = sq_err.mean(dim=(0, 1))  # (D,)
    rmse_per_dim = torch.sqrt(mse_per_dim)

    return rmse_per_sample, rmse_overall, rmse_per_dim


def one_step_rmse_observation(
    y_true_next: torch.Tensor,
    y_pred_samples: torch.Tensor,
):
    """
    Compute 1-step-ahead RMSE in observation space.

    Args:
        y_true_next:    (N, D) tensor of true next-step observations.
                        N = number of synthetic test cases, D = obs_dim.
        y_pred_samples: (N, K, D) tensor of predicted next-step observations
                        from the posterior predictive, where
                        K = number of posterior samples used per case.

    We first average over K to get the mean one-step prediction per case,
    then compute RMSE over all cases and per dimension.

    Returns:
        rmse_overall: scalar float (Python) with overall RMSE.
        rmse_per_dim: 1D numpy array of shape (D,) with per-dim RMSE.
    """
    assert y_true_next.ndim == 2, f"y_true_next must be (N,D), got {y_true_next.shape}"
    assert (
        y_pred_samples.ndim == 3
    ), f"y_pred_samples must be (N,K,D), got {y_pred_samples.shape}"

    N, D = y_true_next.shape
    N2, K, D2 = y_pred_samples.shape
    assert N == N2 and D == D2, "Shape mismatch between y_true_next and y_pred_samples."

    # Mean prediction over posterior samples K
    y_pred_mean = y_pred_samples.mean(dim=1)  # (N, D)

    sq_err = (y_pred_mean - y_true_next) ** 2  # (N, D)

    # Overall RMSE
    rmse_overall = torch.sqrt(sq_err.mean())  # scalar

    # Per-dim RMSE
    rmse_per_dim = torch.sqrt(sq_err.mean(dim=0))  # (D,)

    return float(rmse_overall), rmse_per_dim.cpu().numpy()


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
    _, rmse_overall, rmse_per_dim_t = rmse_posterior_predictive(y_real_t, y_ppc_t)
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


def wasserstein2_posterior_vs_true(
    theta_true: torch.Tensor,
    theta_samples: torch.Tensor,
    num_projections: int = 1000,
    seed: int = 0,
) -> dict:
    """
    Compute Wasserstein-2 style metrics between posterior samples and true theta.

    This measures how well the posterior captures the true parameter value.
    For simulation-based experiments where we know θ_true.

    Args:
        theta_true: (N, d) true parameter values
        theta_samples: (N, K, d) posterior samples for each observation
        num_projections: number of projections for sliced W2
        seed: random seed for projections

    Returns:
        dict with:
          - w2_mean: average W2 distance to true theta
          - w2_per_dim: (d,) per-dimension W2 distances
          - coverage_90: fraction of true thetas within 90% credible region
          - coverage_50: fraction of true thetas within 50% credible region
    """
    assert theta_true.ndim == 2
    assert theta_samples.ndim == 3
    N, K, d = theta_samples.shape
    assert theta_true.shape == (N, d)

    # Convert to numpy
    theta_true_np = theta_true.detach().cpu().numpy()
    theta_samples_np = theta_samples.detach().cpu().numpy()

    # 1) Proper W2 distance to point-mass ground truth per case.
    # For each case i: W2(q_i, delta_theta*) = sqrt(E_q ||theta - theta*||^2)
    sq_dist = np.sum(
        (theta_samples_np - theta_true_np[:, None, :]) ** 2, axis=2
    )  # (N, K)
    w2_per_case = np.sqrt(np.mean(sq_dist, axis=1))  # (N,)

    # Keep posterior-mean error as complementary point-estimate metric.
    posterior_mean = theta_samples_np.mean(axis=1)  # (N, d)
    l2_errors = np.sqrt(np.sum((posterior_mean - theta_true_np) ** 2, axis=1))  # (N,)

    # 2. Per-dimension Wasserstein-1 (sorted quantile matching)
    # For each dimension, compare sorted posterior samples to replicated true value
    w1_per_dim = np.zeros(d)
    for dim in range(d):
        w1_vals = []
        for i in range(N):
            samples_sorted = np.sort(theta_samples_np[i, :, dim])
            # W1 to point mass at true value = mean |sample - true|
            w1_vals.append(np.mean(np.abs(samples_sorted - theta_true_np[i, dim])))
        w1_per_dim[dim] = np.mean(w1_vals)

    # 3) Coverage metrics (does true value fall within credible interval?)
    coverage_90 = _compute_coverage(theta_true_np, theta_samples_np, level=0.90)
    coverage_50 = _compute_coverage(theta_true_np, theta_samples_np, level=0.50)

    # Simformer-style expected coverage curve: empirical coverage vs nominal alpha.
    coverage_alphas = np.linspace(0.05, 0.95, 19)
    coverage_empirical = np.array(
        [
            _compute_coverage(theta_true_np, theta_samples_np, level=float(a))
            for a in coverage_alphas
        ]
    )
    coverage_curve_mae = float(np.mean(np.abs(coverage_empirical - coverage_alphas)))

    # 4. Sliced Wasserstein between aggregated posteriors and true point mass
    # (This gives a distributional distance measure)
    rng = jax.random.PRNGKey(seed)
    all_posterior_samples = theta_samples_np.reshape(-1, d)  # (N*K, d)
    # Replicate true thetas K times to match
    true_replicated = np.repeat(theta_true_np, K, axis=0)  # (N*K, d)
    swd = sliced_wasserstein_distance(
        rng,
        jnp.array(true_replicated),
        jnp.array(all_posterior_samples),
        num_projections=num_projections,
    )

    return {
        "w2_mean": float(np.mean(w2_per_case)),
        "w2_std": float(np.std(w2_per_case)),
        "w1_per_dim": w1_per_dim.tolist(),
        "swd_posterior_vs_true": float(swd),
        "coverage_90": float(coverage_90),
        "coverage_50": float(coverage_50),
        "coverage_curve_alpha": [float(a) for a in coverage_alphas],
        "coverage_curve_empirical": [float(c) for c in coverage_empirical],
        "coverage_curve_mae": coverage_curve_mae,
        "l2_error_mean": float(np.mean(l2_errors)),
        "l2_error_std": float(np.std(l2_errors)),
    }


def _compute_coverage(
    theta_true: np.ndarray, theta_samples: np.ndarray, level: float
) -> float:
    """
    Compute empirical coverage: fraction of true values within credible interval.

    Args:
        theta_true: (N, d) true values
        theta_samples: (N, K, d) posterior samples
        level: credible level (e.g., 0.90 for 90%)

    Returns:
        Fraction of true values covered (averaged over dimensions)
    """
    N, K, d = theta_samples.shape
    alpha = 1 - level
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    covered_per_dim = np.zeros(d)
    for dim in range(d):
        lower = np.quantile(theta_samples[:, :, dim], lower_q, axis=1)  # (N,)
        upper = np.quantile(theta_samples[:, :, dim], upper_q, axis=1)  # (N,)
        covered = (theta_true[:, dim] >= lower) & (theta_true[:, dim] <= upper)
        covered_per_dim[dim] = np.mean(covered)

    return np.mean(covered_per_dim)
