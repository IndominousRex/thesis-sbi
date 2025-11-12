---
type: plan
title: "Implementation Plan"
status: in-progress
tags: [thesis, implementation]
---

# 🧠 Implementation Plan

## 1️⃣ Core Modules

| Module | Purpose | Notes |
|--------|----------|-------|
| `simulation/vehicle_model.py` | JAX-based simulator wrapper | Deterministic and differentiable |
| `simulation/generate_dataset.py` | Generate (θ, y) pairs | Use PRBS and sine-sweep control inputs |
| `inference/npe_train.py` | Neural Posterior Estimation | sbi-based or custom GRU-encoder NPE |
| `inference/npse_train.py` | Neural Posterior *Score* Estimation | Diffusion- or score-based model |
| `utils/metrics.py` | Evaluation metrics | RMSE, W₂, coverage, runtime |
| `utils/plotting.py` | Result visualizations | Posterior and calibration plots |

---

## 2️⃣ Key Interfaces & Data Shapes

| Function | Input → Output | Description |
|-----------|----------------|-------------|
| `simulate_y_batch(params, controls)` | θ ∈ ℝᵖ → y ∈ ℝᴮˣᵀˣᴰ | Runs simulator for a batch of parameters |
| `train_density_estimator(model, data, config)` | dataset → trained model | Unified training entry point |
| `evaluate_posterior(posterior, θ_true, y_obs)` | → dict(metrics) | Computes RMSE, W₂, coverage |
| `save_results(config, metrics, plots)` | — | Logs to `experiments/results/` |

**Notation:**  
- \(B\): batch size \(T\): time-steps \(D\): observation dimension

---

## 3️⃣ Data & Configuration Conventions

- **Simulation data:** `experiments/data/{sim_name}/{run_id}/`
- **Configs:** YAML files under `code/configs/`
- **Naming:** `YYYYMMDD_method_experiment`
- **Random seeds:** 42 by default; store in config.

---

## 4️⃣ Evaluation Metrics

| Metric | Formula / Idea | Purpose |
|---------|----------------|---------|
| **RMSE** | \( \sqrt{\frac{1}{p}\sum_i(\hat{\theta}_i-\theta_i)^2} \) | Parameter estimation accuracy |
| **W₂ distance** | Wasserstein-2 between posterior samples | Posterior shape similarity |
| **Coverage** | % of θ true within credible interval | Calibration quality |
| **Runtime** | sec/epoch or total training time | Computational efficiency |

---

## 5️⃣ Reproducibility Notes

- Fix seeds for NumPy, Torch, JAX.  
- Save all configs + model checkpoints + plots under each `experiments/` folder.  
- Export `requirements.txt` after environment changes.  
- Use Git tags for milestone commits (`v0.1-npe-baseline` etc.).
