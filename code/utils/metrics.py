import torch
import numpy as np
from scipy.stats import wasserstein_distance

def rmse(theta_hat, theta_true):
    """Root Mean Squared Error."""
    return torch.sqrt(torch.mean((theta_hat - theta_true) ** 2)).item()

def w2_distance(samples1, samples2):
    """Approximate Wasserstein-2 distance for 1D arrays."""
    s1, s2 = samples1.detach().cpu().numpy(), samples2.detach().cpu().numpy()
    return wasserstein_distance(s1.flatten(), s2.flatten())

def coverage(posterior_samples, theta_true, ci=0.9):
    """Fraction of true parameters within credible interval."""
    lower = np.quantile(posterior_samples, (1-ci)/2, axis=0)
    upper = np.quantile(posterior_samples, 1-(1-ci)/2, axis=0)
    inside = np.logical_and(theta_true >= lower, theta_true <= upper)
    return np.mean(inside)

def runtime(start_time, end_time):
    """Total elapsed time in seconds."""
    return end_time - start_time
