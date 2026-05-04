# Simulation-Based Inference for Vehicle Parameter Estimation

**Master's Thesis — Aritra Saha (10065012)**
**Institute of Mechatronic Systems (IMES), Leibniz Universität Hannover**
**Supervisor: Jan-Hendrik Ewering | Duration: Nov 2025 – Apr 2026**

---

This repository contains the full implementation, experiments, and thesis text for the master's thesis *"Simulation-based Inference for Parameter Density Estimation in Dynamical Systems"*.

The core research question: **Can amortized simulation-based inference (SBI) efficiently and accurately estimate physical vehicle parameters — friction coefficient (µ), air resistance (cd), and mass (m) — from time-series driving data?**

Four neural posterior estimation methods are implemented and benchmarked on a JAX two-track vehicle dynamics simulator and real measurement data:
- **NPE** — Neural Posterior Estimation with normalizing flows (MAF)
- **NPSE** — Neural Posterior Score Estimation with diffusion models
- **FNPE** — Factorized NPE via the MarkovSBI framework (JAX) — [Gloeckler et al., 2024](https://arxiv.org/abs/2411.02728)
- **Simformer** — All-conditional score transformer — [Gloeckler et al., 2024](https://arxiv.org/abs/2404.09636)

---

## Table of Contents

- [Simulation-Based Inference for Vehicle Parameter Estimation](#simulation-based-inference-for-vehicle-parameter-estimation)
  - [Table of Contents](#table-of-contents)
  - [Repository Structure](#repository-structure)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Environment](#environment)
    - [Vendored Libraries](#vendored-libraries)
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
  - [Thesis Text](#thesis-text)
    - [Chapters](#chapters)
    - [Building the PDF](#building-the-pdf)
    - [Thesis Figures](#thesis-figures)
  - [Notebooks](#notebooks)
  - [Cluster (SLURM)](#cluster-slurm)
    - [Script Reference](#script-reference)
    - [Typical Benchmark Workflow](#typical-benchmark-workflow)
  - [Development Notes](#development-notes)
    - [Code Architecture](#code-architecture)
    - [Prior](#prior)
    - [Dataset Caching](#dataset-caching)
    - [FNPE Score Compositions](#fnpe-score-compositions)
  - [References](#references)

---

## Repository Structure

```
thesis-sbi-aritra/
│
├── code/                        # All Python code
│   ├── run.py                   # Main CLI entry point
│   ├── run_fnpe_pso.py          # FNPE with PSO-calibrated vehicle parameters
│   │
│   ├── configs/
│   │   └── config.py            # ExperimentConfig dataclass (~150 fields)
│   │
│   ├── inference/
│   │   └── unified_experiment.py  # Full train→eval pipeline (run_experiment)
│   │
│   ├── methods/
│   │   ├── base.py              # Abstract BaseMethod interface
│   │   ├── npe_method.py        # NPE (MAF + BiGRU, via sbi library)
│   │   ├── npse_method.py       # NPSE (diffusion, via sbi library)
│   │   ├── fnpe_method.py       # FNPE (factorized score, via MarkovSBI/JAX)
│   │   └── simformer_method.py  # Simformer (all-conditional transformer, JAX)
│   │
│   ├── models/
│   │   └── models.py            # build_prior(), build_embedding(), encoder nets
│   │
│   ├── simulation/
│   │   ├── VehicleModel.py      # JAX two-track dynamics (Magic Tire Formula, RK4)
│   │   ├── simulation.py        # make_simulator(), dataset generation pipeline
│   │   ├── noise.py             # Calibrated obs + process noise injection
│   │   └── utils.py             # WGS-84 and rotation matrix utilities
│   │
│   ├── utils/
│   │   ├── normalization.py     # Normalizer dataclass and fit_normalizer()
│   │   ├── metrics.py           # RMSE, W2, SWD, coverage, C2ST metrics
│   │   ├── plots.py             # Diagnostic plot functions
│   │   ├── real_data.py         # Real CSV loading and PPC evaluation
│   │   └── env_utils.py         # Device setup and seed initialization
│   │
│   ├── scripts/
│   │   ├── run_simulation_comparison.py     # Multi-method comparison runner
│   │   ├── aggregate_simulation_benchmark.py # Aggregate results into CSV/plots
│   │   ├── submit_simulation_benchmark_array.py # Build + submit SLURM manifests
│   │   ├── launch_simulation_benchmark.py   # Local or SLURM benchmark launcher
│   │   ├── rerun_experiment_from_config.py  # Fresh rerun from saved config.json
│   │   ├── recover_saved_eval.py            # Recover eval metrics from checkpoints
│   │   ├── run_pso_global.py                # PSO vehicle parameter optimization
│   │   ├── plot_pso_convergence.py          # PSO convergence figure (thesis)
│   │   ├── plot_pso_trajectory_comparison.py # Default vs PSO trajectory figure
│   │   ├── plot_training_data_example.py    # Training data example figure (thesis)
│   │   ├── plot_cover_image.py              # Thesis cover image (prior-to-posterior flow)
│   │   ├── run_simulation_benchmark_manifest_cell.py # Single SLURM array cell executor
│   │   └── *.sh                             # SLURM batch scripts
│   │
│   ├── notebooks/               # Jupyter notebooks for exploration
│   ├── markovsbi/               # MarkovSBI library (FNPE, vendored)
│   ├── simformer-main/          # Simformer library (vendored)
│   ├── experiments/             # Output directory for all experiment results
│   └── tmp/                     # Temporary/one-off analysis scripts
│
├── data/
│   └── measurements/            # 7 real vehicle measurement CSVs
│
├── text (Latex)/                # Thesis LaTeX source
│   ├── main.tex                 # Master document
│   ├── Chapters/                # Individual chapter .tex files
│   ├── Figures/                 # Figures used in the thesis
│   └── Templates/               # Bibliography, preamble, nomenclature
│
├── presentations/               # Seminar and progress slides
└── admin/                       # Task description PDF and reference thesis
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA 12.2+ (for GPU acceleration; CPU fallback supported)
- Conda (Miniforge recommended)

### Environment

The project runs in a shared conda environment on the Leibniz University cluster:

```bash
# On the cluster
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe
```

For a local setup, the key packages are:

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

### Vendored Libraries

Two external libraries are vendored under `code/` and must be on the Python path:

| Library   | Path                       | Purpose                           | Paper                                                      | GitHub                                                      |
| --------- | -------------------------- | --------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------- |
| MarkovSBI | `code/markovsbi/`          | FNPE factorized score estimation  | [Gloeckler et al., 2024](https://arxiv.org/abs/2411.02728) | [mackelab/markovsbi](https://github.com/mackelab/markovsbi) |
| Simformer | `code/simformer-main/src/` | All-conditional score transformer | [Gloeckler et al., 2024](https://arxiv.org/abs/2404.09636) | [mackelab/simformer](https://github.com/mackelab/simformer) |

The `simformer_method.py` adds these to `sys.path` automatically at import time.

### Environment Variables

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # Prevent JAX from grabbing all VRAM
export JAX_PLATFORM_NAME=gpu                  # Set to "cpu" if no GPU
export OMP_NUM_THREADS=8                      # Match SLURM cpus-per-task
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
| --------- | ---------------------- | ---------------------- | ----------------------------------------- |
| Core      | `--method`             | `npe`                  | `npe`, `npse`, `fnpe`, `simformer`        |
| Core      | `--params`             | `mu,cd,m`              | Comma-separated parameters to infer       |
| Core      | `--device`             | `cuda`                 | `cuda`, `cpu`, `auto`                     |
| Data      | `--num-sim`            | `2000`                 | Number of training trajectories           |
| Data      | `--T-seg`              | `3000`                 | Trajectory length (timesteps at 100 Hz)   |
| Data      | `--seed`               | `42`                   | Base random seed                          |
| Training  | `--num-epochs`         | `300`                  | Training epochs (NPE/NPSE)                |
| Training  | `--lr`                 | `5e-4`                 | Learning rate                             |
| Training  | `--batch-size`         | `512`                  | Training batch size                       |
| Encoder   | `--encoder-type`       | `bigru`                | `bigru`, `causalcnn`, `transformer`       |
| Encoder   | `--encoder-hidden`     | `32`                   | Hidden units in encoder                   |
| NPE       | `--maf-hidden`         | `128`                  | MAF hidden features                       |
| NPE       | `--maf-transforms`     | `8`                    | Number of MAF transforms                  |
| NPSE      | `--sde-type`           | `ve`                   | `ve` (SMLD), `vp` (DDPM), `subvp`         |
| FNPE      | `--fnpe-num-sim`       | `100000`               | FNPE simulation budget                    |
| FNPE      | `--fnpe-score-fn`      | `gauss_corrected`      | Score composition type                    |
| FNPE      | `--fnpe-proposal-type` | `pred`                 | Proposal strategy for training data       |
| FNPE      | `--fnpe-window-size`   | `2`                    | Markov window size                        |
| Eval      | `--num-sbc-samples`    | `200`                  | Samples for SBC diagnostics               |
| Eval      | `--no-sbc`             | —                      | Disable SBC                               |
| Eval      | `--no-one-step`        | —                      | Disable one-step RMSE                     |
| Real data | `--real-data-dir`      | `../data/measurements` | Directory of measurement CSVs             |
| Real data | `--k-ppc`              | `300`                  | Posterior predictive samples per PPC plot |
| Output    | `--results-root`       | `experiments`          | Parent directory for output               |

---

## Evaluation Metrics

Each completed experiment writes a `results.json` with:

| Metric                  | Description                                                     |
| ----------------------- | --------------------------------------------------------------- |
| `w2_mean`               | Mean Wasserstein-2 distance between posterior and true θ        |
| `coverage_90_phys`      | 90% credible interval empirical coverage (physical space)       |
| `coverage_abs_error`    | Mean absolute deviation of coverage from nominal levels         |
| `c2st_mean`             | C2ST accuracy (0.5 = perfect, 1.0 = completely wrong)           |
| `one_step_rmse`         | 1-step-ahead observation RMSE from posterior predictive         |
| `swd_posterior_vs_true` | Sliced Wasserstein distance, posterior samples vs. true θ       |
| `real_ppc_w2`           | W2 between posterior predictive and real measurement trajectory |
| `train_time_s`          | Wall-clock training time in seconds                             |
| Per-parameter           | `mu_bias`, `mu_rmse`, `mu_coverage_90`, `cd_*`, `m_*`           |

---

## Real Measurement Data

Seven real vehicle recordings are in `data/measurements/`:

| File                                                  | Description                                |
| ----------------------------------------------------- | ------------------------------------------ |
| `Jeversen_2021_12_15_112145_Asphalt-Vollbremsung.csv` | Asphalt full braking #1                    |
| `Jeversen_2021_12_15_112328_Asphalt-Vollbremsung.csv` | Asphalt full braking #2                    |
| `Jeversen_2021_12_15_125650_Beton-Vollbremsung.csv`   | Concrete full braking #1                   |
| `Jeversen_2021_12_15_125936_Beton-Vollbremsung.csv`   | Concrete full braking #2                   |
| `Jeversen_2021_12_15_134709_Basalt-Vollbremsung.csv`  | Basalt full braking #1                     |
| `Jeversen_2021_12_15_134858_Basalt-Vollbremsung.csv`  | Basalt full braking #2                     |
| `Jeversen_2022_10_12_110132.csv`                      | Mixed driving (used for training-mode PPC) |

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

## Thesis Text

The LaTeX source is in `text (Latex)/`. The main document is [`text (Latex)/main.tex`](text%20(Latex)/main.tex).

### Chapters

| File                                         | Content                                                 |
| -------------------------------------------- | ------------------------------------------------------- |
| `introduction.tex`                           | Motivation, research questions, contributions           |
| `related_work.tex`                           | SBI landscape, NPE, NPSE, FNPE, Simformer, research gap |
| `problem_setting_and_vehicle_case_study.tex` | Vehicle model, Magic Tire Formula, noise model, priors  |
| `implemented_methods.tex`                    | NPE, NPSE, FNPE, Simformer architectures and training   |
| `comparative_evaluation.tex`                 | Benchmark results and analysis                          |
| `conclusion_and_outlook.tex`                 | Conclusions and future work                             |

### Building the PDF

```bash
cd "text (Latex)"
latexmk -pdf -interaction=nonstopmode main.tex
```

Requires a LaTeX distribution (TeX Live 2022+ recommended) with `latexmk`.

### Thesis Figures

Generated figures for the thesis are saved to `text (Latex)/Figures/` by the plotting scripts in `code/scripts/`. Key figure scripts:

```bash
# PSO convergence history
python code/scripts/plot_pso_convergence.py

# Default vs. PSO-calibrated simulator trajectory comparison
python code/scripts/plot_pso_trajectory_comparison.py

# Training data example (µ ∈ {0.6, 0.9, 1.4})
python code/scripts/plot_training_data_example.py

# Thesis cover image (prior-to-posterior density flow)
python code/scripts/plot_cover_image.py
```

---

## Notebooks

Exploratory Jupyter notebooks are in `code/notebooks/`:

| Notebook                                  | Purpose                                            |
| ----------------------------------------- | -------------------------------------------------- |
| `NPE.ipynb`                               | Early NPE prototype and exploration                |
| `data_distribution_analysis.ipynb`        | Dataset statistics and prior coverage analysis     |
| `pso_trajectory_evaluation.ipynb`         | PSO optimization results and trajectory comparison |
| `realistic_simulation_improvements.ipynb` | Noise calibration from real measurement residuals  |
| `simulation_benchmark_validation.ipynb`   | Benchmark result validation                        |

Run from the `code/` directory so relative imports resolve correctly.

---

## Cluster (SLURM)

All Slurm scripts are in `code/scripts/`. They are written for the **Leibniz University HPC cluster** (`login.cluster.uni-hannover.de`) with:
- Partition: `gpu`
- Environment path: `/software/NHKB22930/nhkbarit/conda_envs/npe`
- H200 GPU resource: `--gres=gpu:h200:1`

### Script Reference

| Script                              | Purpose                                      |
| ----------------------------------- | -------------------------------------------- |
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
squeue -u nhkbarit

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

Default bounds: µ ∈ [0.5, 1.5], cd ∈ [0.05, 0.6], m ∈ [1700, 2200] kg.

### Dataset Caching

Datasets are cached by content hash (deterministic from config fields) under `code/datasets/`. Use `--reuse-dataset` to reload an existing dataset and `--no-cache` to disable caching entirely. In benchmark runs, caching is disabled by default so every cell regenerates data deterministically from its seed.

### FNPE Score Compositions

| Value             | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| `gauss_corrected` | Precision-weighted Gaussian correction (default, most accurate) |
| `fnpe`            | Standard FNPE score composition (faster)                        |
| `uncorrected`     | No prior correction (for ablation)                              |

---

## References

This project builds on the following external libraries and their associated papers:

**MarkovSBI (FNPE)**
> Manuel Gloeckler, Shoji Toyota, Kenji Fukumizu, Jakob H. Macke.  
> *Compositional simulation-based inference for time series.*  
> arXiv:2411.02728, 2024.  
> Paper: https://arxiv.org/abs/2411.02728  
> Code: https://github.com/mackelab/markovsbi

**Simformer**
> Manuel Gloeckler, Michael Deistler, Christian Weilbach, Frank Wood, Jakob H. Macke.  
> *All-in-one simulation-based inference.*  
> arXiv:2404.09636, 2024.  
> Paper: https://arxiv.org/abs/2404.09636  
> Code: https://github.com/mackelab/simformer