# Tutorial: Setup, Running, and Reproducing Results

Detailed reference for setting up the environment, running experiments, and reproducing benchmark results. See [README.md](README.md) for the project overview.

## Table of Contents

- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Environment](#environment)
  - [External Dependencies](#external-dependencies)
  - [Environment Variables](#environment-variables)
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
  - [Single Method Run](#single-method-run)
  - [Multi-Method Comparison](#multi-method-comparison)
  - [Benchmark Grid (SLURM)](#benchmark-grid-slurm)
  - [Eval-Only from Checkpoint](#eval-only-from-checkpoint)
  - [PSO-Calibrated FNPE](#pso-calibrated-fnpe)
- [Key CLI Arguments](#key-cli-arguments)
- [Evaluation Metrics](#evaluation-metrics)
- [Real Measurement Data](#real-measurement-data)
- [Aggregating Benchmark Results](#aggregating-benchmark-results)
- [Notebooks](#notebooks)
- [Cluster (SLURM) Notes](#cluster-slurm-notes)
- [Development Notes](#development-notes)

---

## Setup

### Getting the Code

```bash
git clone https://github.com/IndominousRex/thesis-sbi.git
cd thesis-sbi
```

Everything under `code/` is a normal Python project — no build step required beyond the environment setup below. All commands in this tutorial assume you're running from inside the cloned repo, with your working directory set to `code/` unless stated otherwise.

### Prerequisites

- Python 3.10+
- CUDA 12.2+ (for GPU acceleration; CPU fallback supported)
- Conda (Miniforge recommended)

### Environment

```bash
# PyTorch stack
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# SBI library
pip install sbi

# JAX with GPU support
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Other dependencies
pip install numpy scipy pandas matplotlib tqdm optax haiku dm-haiku
```

If running on a shared institutional cluster, use whatever pre-built environment your cluster provides, or build the one above from scratch in your own environment path.

### External Dependencies

FNPE and Simformer rely on two external libraries that are **not vendored in this repo** — clone them separately and add to your Python path:

| Library   | Purpose                          | Paper                                                      | Code                                                          |
| --------- | --------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------- |
| MarkovSBI | FNPE factorized score estimation | [Gloeckler et al., 2024](https://arxiv.org/abs/2411.02728) | [mackelab/markovsbi](https://github.com/mackelab/markovsbi) |
| Simformer | All-conditional score transformer | [Gloeckler et al., 2024](https://arxiv.org/abs/2404.09636) | [mackelab/simformer](https://github.com/mackelab/simformer) |

```bash
git clone https://github.com/mackelab/markovsbi.git
git clone https://github.com/mackelab/simformer.git
# Add both to PYTHONPATH, or place them alongside code/ and adjust sys.path in simformer_method.py
```

### Environment Variables

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # Prevent JAX from grabbing all VRAM
export JAX_PLATFORM_NAME=gpu                  # Set to "cpu" if no GPU
export OMP_NUM_THREADS=8                      # Match available CPU cores / SLURM cpus-per-task
```

---

## Quick Start

All experiments are launched from the `code/` directory.

```bash
cd code/

# Minimal smoke test — all methods, 100 simulations, 300 timesteps
python scripts/run_simulation_comparison.py \
    --exp-name smoke_test \
    --num-simulations 100 \
    --T-seg 300 \
    --methods npe npse \
    --smoke

# Results appear in code/experiments/smoke_test_*/
```

---

## Running Experiments

### Single Method Run

Use `run.py` to train and evaluate a single method:

```bash
# NPE — friction only, 5000 simulations, 3000 timesteps
python run.py \
    --method npe \
    --exp-name npe_mu_baseline \
    --params mu \
    --num-sim 5000 \
    --T-seg 3000 \
    --num-epochs 200 \
    --seed 42

# NPSE — all three parameters
python run.py \
    --method npse \
    --exp-name npse_all_params \
    --params mu,cd,m \
    --num-sim 5000 \
    --T-seg 3000 \
    --sde-type ve

# FNPE — factorized score, gauss-corrected composition
python run.py \
    --method fnpe \
    --exp-name fnpe_mu_gauss \
    --params mu \
    --fnpe-num-sim 100000 \
    --fnpe-score-fn gauss_corrected \
    --T-seg 3000

# Simformer — all-conditional transformer
python run.py \
    --method simformer \
    --exp-name simformer_mu \
    --params mu \
    --num-sim 5000 \
    --T-seg 1000
```

Results are written to `experiments/<exp-name>_<method>_<timestamp>/` and include:
- `config.json` — full experiment configuration
- `results.json` — all evaluation metrics
- `figures/` — diagnostic plots (posterior, SBC, PPC, training curves)
- `model.*` — saved model checkpoint

### Multi-Method Comparison

Run all methods on the same dataset:

```bash
python scripts/run_simulation_comparison.py \
    --exp-name bench_v1 \
    --num-simulations 5000 \
    --T-seg 1000 \
    --methods npe npse fnpe simformer \
    --params mu,cd,m \
    --seed 42
```

Use `--requested-budget-steps` to normalize across methods by total simulator steps rather than number of trajectories:

```bash
python scripts/run_simulation_comparison.py \
    --exp-name bench_budget \
    --requested-budget-steps 20000000 \
    --T-seg 1000 \
    --methods npe npse fnpe simformer
```

### Benchmark Grid (SLURM)

Generate and submit a full sweep over methods × parameter sets × sequence lengths × seeds × budgets:

```bash
cd code/

# Build the manifest and submit to SLURM
python scripts/submit_simulation_benchmark_array.py \
    --group-name bench_budget_v3 \
    --budget-grid 20000000 40000000 60000000 \
    --methods npe npse fnpe simformer \
    --params-grid "mu" "mu,cd" "mu,m" "mu,cd,m" \
    --tseg-grid 1000 2000 3000 \
    --seeds 42 43 44 45 46 \
    --mode slurm-submit \
    --max-concurrent 12

# Or just print the manifest without submitting
python scripts/submit_simulation_benchmark_array.py \
    --group-name bench_budget_v3 \
    --budget-grid 20000000 40000000 \
    --methods npe npse \
    --params-grid "mu" \
    --tseg-grid 1000 \
    --seeds 42 43 \
    --mode print
```

Each cell in the grid creates one SLURM array task. H200 GPUs are requested automatically for NPE, NPSE, and Simformer (full-trajectory methods); FNPE uses any available GPU.

### Eval-Only from Checkpoint

Reload a trained model and re-run evaluation only:

```bash
python run.py \
    --eval \
    --checkpoint experiments/npe_baseline_20260401-120000/ \
    --exp-name npe_eval_rerun \
    --device cuda
```

Or rerun a full fresh train+eval from a saved config (no checkpoint reuse):

```bash
python scripts/rerun_experiment_from_config.py \
    --config experiments/npe_baseline_20260401-120000/config.json \
    --device cuda
```

### PSO-Calibrated FNPE

Run FNPE with vehicle model parameters calibrated via Particle Swarm Optimization (replaces default simulator defaults):

```bash
python run_fnpe_pso.py \
    --pso-json notebooks/experiments/pso_global_log_optimization_results.json \
    --exp-name fnpe_pso_calibrated \
    --fnpe-num-sim 100000 \
    --T-seg 3000
```

---

## Key CLI Arguments

All arguments for `run.py` (full list: `python run.py --help`):

| Group     | Argument               | Default                | Description                               |
| --------- | ----------------------- | ----------------------- | ------------------------------------------ |
| Core      | `--method`              | `npe`                   | `npe`, `npse`, `fnpe`, `simformer`         |
| Core      | `--params`              | `mu,cd,m`                | Comma-separated parameters to infer        |
| Core      | `--device`              | `cuda`                  | `cuda`, `cpu`, `auto`                      |
| Data      | `--num-sim`             | `2000`                  | Number of training trajectories            |
| Data      | `--T-seg`               | `3000`                  | Trajectory length (timesteps at 100 Hz)    |
| Data      | `--seed`                | `42`                    | Base random seed                           |
| Training  | `--num-epochs`          | `300`                   | Training epochs (NPE/NPSE)                 |
| Training  | `--lr`                  | `5e-4`                  | Learning rate                              |
| Training  | `--batch-size`          | `512`                   | Training batch size                        |
| Encoder   | `--encoder-type`        | `bigru`                 | `bigru`, `causalcnn`, `transformer`        |
| Encoder   | `--encoder-hidden`      | `32`                    | Hidden units in encoder                    |
| NPE       | `--maf-hidden`          | `128`                   | MAF hidden features                        |
| NPE       | `--maf-transforms`      | `8`                     | Number of MAF transforms                   |
| NPSE      | `--sde-type`            | `ve`                    | `ve` (SMLD), `vp` (DDPM), `subvp`          |
| FNPE      | `--fnpe-num-sim`        | `100000`                | FNPE simulation budget                     |
| FNPE      | `--fnpe-score-fn`       | `gauss_corrected`       | Score composition type                     |
| FNPE      | `--fnpe-proposal-type`  | `pred`                  | Proposal strategy for training data        |
| FNPE      | `--fnpe-window-size`    | `2`                     | Markov window size                         |
| Eval      | `--num-sbc-samples`     | `200`                   | Samples for SBC diagnostics                |
| Eval      | `--no-sbc`              | —                       | Disable SBC                                |
| Eval      | `--no-one-step`         | —                       | Disable one-step RMSE                      |
| Real data | `--real-data-dir`       | `../data/measurements`  | Directory of measurement CSVs              |
| Real data | `--k-ppc`               | `300`                   | Posterior predictive samples per PPC plot  |
| Output    | `--results-root`        | `experiments`           | Parent directory for output                |

---

## Evaluation Metrics

Each completed experiment writes a `results.json` with:

| Metric                  | Description                                                     |
| ------------------------ | ------------------------------------------------------------------ |
| `w2_mean`                | Mean Wasserstein-2 distance between posterior and true θ         |
| `coverage_90_phys`       | 90% credible interval empirical coverage (physical space)        |
| `coverage_abs_error`     | Mean absolute deviation of coverage from nominal levels          |
| `c2st_mean`              | C2ST accuracy (0.5 = perfect, 1.0 = completely wrong)             |
| `one_step_rmse`          | 1-step-ahead observation RMSE from posterior predictive           |
| `swd_posterior_vs_true`  | Sliced Wasserstein distance, posterior samples vs. true θ         |
| `real_ppc_w2`            | W2 between posterior predictive and real measurement trajectory  |
| `train_time_s`           | Wall-clock training time in seconds                               |
| Per-parameter            | `mu_bias`, `mu_rmse`, `mu_coverage_90`, `cd_*`, `m_*`             |

---

## Real Measurement Data

Seven real vehicle recordings are in `data/measurements/`, covering full braking on three surface types plus a mixed-driving recording:

- Asphalt full braking (2 recordings)
- Concrete full braking (2 recordings)
- Basalt full braking (2 recordings)
- Mixed driving (1 recording, used for training-mode PPC)

**CSV columns used:** `Rate_Body_Z`, `INS_Vel_Body_X`, `INS_Vel_Body_Y`, `Acc_Body_X`, `Acc_Body_Y`, `Tire_Rate_FL/FR/RL/RR`, `Steer_Angle`, `Engine_Torque`, `Break_Pressure`, `Gear_Transmission`.

To run with real-data PPC evaluation, pass `--real-data-dir ../data/measurements` (default) or point to a specific CSV with `--real-data-csv`.

---

## Aggregating Benchmark Results

After a benchmark run, aggregate all experiment folders into summary CSVs and plots:

```bash
cd code/

# Local
python scripts/aggregate_simulation_benchmark.py \
    --experiments-root experiments \
    --exp-prefix bench_budget_v3

# Via SLURM
sbatch scripts/run_aggregate_benchmark.sh bench_budget_v3
```

Outputs (written next to the experiments folder):
- `<prefix>_aggregate_metrics.csv` — one row per experiment cell
- `<prefix>_aggregate_metrics.json` — same + raw rows
- `<prefix>_aggregate_summary.json` — per-method/budget summary statistics
- `figures/<prefix>/` — comparison plots: method dashboard, radar chart, convergence profile, training time scaling, win-rate heatmap, T_seg scaling, and more

---

## Notebooks

Exploratory Jupyter notebooks are in `code/notebooks/`:

| Notebook                                  | Purpose                                            |
| ------------------------------------------- | ----------------------------------------------------- |
| `NPE.ipynb`                               | Early NPE prototype and exploration                |
| `data_distribution_analysis.ipynb`        | Dataset statistics and prior coverage analysis     |
| `pso_trajectory_evaluation.ipynb`         | PSO optimization results and trajectory comparison |
| `realistic_simulation_improvements.ipynb` | Noise calibration from real measurement residuals  |
| `simulation_benchmark_validation.ipynb`   | Benchmark result validation                        |

Run from the `code/` directory so relative imports resolve correctly.

---

## Cluster (SLURM) Notes

All SLURM scripts are in `code/scripts/`, written for a SLURM-managed GPU cluster with H200 nodes. Adjust the partition name, environment path, and GPU resource string in each `.sh` script to match your own cluster before submitting.

### Script Reference

| Script                              | Purpose                                      |
| ------------------------------------- | ----------------------------------------------- |
| `run_experiment.sh`                 | Single method, single GPU                    |
| `run_comparison.sh`                 | All 4 methods sequentially on same dataset   |
| `run_simulation_comparison.sh`      | SLURM array: 4 methods × 4 parallel tasks    |
| `run_simulation_single.sh`          | Single cell of a comparison grid             |
| `run_simulation_benchmark_array.sh` | Manifest-driven SLURM array (main benchmark) |
| `run_aggregate_benchmark.sh`        | CPU job to aggregate benchmark results       |
| `run_rerun_from_config_array.sh`    | Rerun failed cells from saved configs        |
| `run_fnpe_ablation.sh`              | FNPE proposal type ablation study            |
| `run_parameter_sweep.sh`            | Vehicle parameter sensitivity sweep          |
| `run_real_data_eval.sh`             | Eval-only with real data (12-task array)     |
| `run_simformer_smoke.sh`            | 30-min Simformer smoke test                  |
| `test_all_methods.sh`               | Quick test of all 4 methods (1 GPU, ~1h)     |
| `run_fnpe_pso.sh`                   | FNPE with PSO-calibrated parameters          |
| `run_pso_global.sh`                 | PSO global vehicle parameter optimisation    |
| `recover_saved_eval.sh`             | GPU array job to recover eval from saved checkpoints |

### Typical Benchmark Workflow

```bash
# 1. Submit benchmark grid
python scripts/submit_simulation_benchmark_array.py \
    --group-name bench_budget_v4 \
    --budget-grid 20000000 40000000 60000000 \
    --mode slurm-submit

# 2. Monitor jobs
squeue -u <your-cluster-username>

# 3. Recover any failed cells
python scripts/recover_saved_eval.py \
    --experiments-dir experiments \
    --match-prefix bench_budget_v4

# 4. Aggregate results
sbatch scripts/run_aggregate_benchmark.sh bench_budget_v4
```

---

## Development Notes

### Code Architecture

The unified pipeline in `inference/unified_experiment.py` → `run_experiment(cfg)` handles all methods identically:

```
theta_phys ─[prior sample]──► normalize ──► train method ──► posterior.sample(x_norm) ──► unnormalize ──► theta_phys
x_phys [obs ‖ ctrl]   ──────► normalize ──► condition posterior
```

All four methods share:
- The same vehicle simulator and dataset generation
- The same normalizer (`utils/normalization.py`)
- The same evaluation suite (W2, C2ST, coverage, one-step RMSE, PPC)
- The same `ExperimentConfig` dataclass

### Prior

Gaussian prior centered at the midpoint of the parameter range, with std = (high − low) / 6 so that ±3σ approximately covers the original uniform bounds:

```python
mean = (low + high) / 2
std  = (high - low) / 6
```

Default bounds: µ ∈ [0.5, 1.5], c_d ∈ [0.05, 0.6], m ∈ [1700, 2200] kg.

### Dataset Caching

Datasets are cached by content hash (deterministic from config fields) under `code/datasets/`. Use `--reuse-dataset` to reload an existing dataset and `--no-cache` to disable caching entirely. In benchmark runs, caching is disabled by default so every cell regenerates data deterministically from its seed.

### FNPE Score Compositions

| Value             | Description                                                     |
| ------------------- | ------------------------------------------------------------------- |
| `gauss_corrected` | Precision-weighted Gaussian correction (default, most accurate) |
| `fnpe`             | Standard FNPE score composition (faster)                        |
| `uncorrected`      | No prior correction (for ablation)                              |
