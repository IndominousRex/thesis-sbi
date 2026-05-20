#!/bin/bash -l
# ==============================================================================
# Slurm CPU job to run plot_defense_figures.py and produce presentation-ready
# figures for the thesis defense (simplified style, large labels/ticks).
#
# Usage (from the code/ directory):
#   sbatch scripts/run_defense_figures.sh <exp_prefix> [experiments_root] [output_dir]
#
# Examples:
#   sbatch scripts/run_defense_figures.sh bench_budget_v3_fixed
#   sbatch scripts/run_defense_figures.sh bench_budget_v3_fixed experiments
#   sbatch scripts/run_defense_figures.sh bench_budget_v3_fixed experiments /path/to/custom/out
#
# Outputs land in:
#   <experiments_root>/<exp_prefix>/plots_defense/
# (or <output_dir>/ if the third argument is supplied)
# ==============================================================================

#SBATCH --job-name=defense_figs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
#SBATCH --output=defense_figs_%j.out
#SBATCH --error=defense_figs_%j.err

set -euo pipefail

EXP_PREFIX=${1:?exp_prefix argument required}
EXPERIMENTS_ROOT=${2:-experiments}
OUTPUT_DIR=${3:-}          # optional; if empty, script uses its own default

# ---------------------------------------------------------------------------
# Locate code root (supports both interactive and sbatch invocation)
# ---------------------------------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/plot_defense_figures.py" ]; then
    CODE_DIR="${SLURM_SUBMIT_DIR}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/plot_defense_figures.py" ]; then
    CODE_DIR=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)
elif [ -d "/bigwork/nhkbarit/thesis-code/code" ]; then
    CODE_DIR="/bigwork/nhkbarit/thesis-code/code"
else
    echo "Could not determine code directory."
    echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
    exit 1
fi

echo "=================================================="
echo "Defense Figure Generator"
echo "=================================================="
echo "CODE_DIR:         ${CODE_DIR}"
echo "EXP_PREFIX:       ${EXP_PREFIX}"
echo "EXPERIMENTS_ROOT: ${EXPERIMENTS_ROOT}"
echo "OUTPUT_DIR:       ${OUTPUT_DIR:-<default: experiments/<prefix>/plots_defense>}"
echo "=================================================="

cd "${CODE_DIR}"

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
module load Miniforge3
ENV_PREFIX="${SBI_ENV_PREFIX:-/software/NHKB22930/nhkbarit/conda_envs/npe}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

unset PYTHONHOME || true
unset PYTHONPATH || true
export PATH="${ENV_PREFIX}/bin:${PATH}"

echo "Python: $(conda run -p "${ENV_PREFIX}" --no-capture-output python --version)"

# ---------------------------------------------------------------------------
# Build optional extra arguments
# ---------------------------------------------------------------------------
EXTRA_ARGS=""
if [ -n "${OUTPUT_DIR}" ]; then
    EXTRA_ARGS="--output-dir ${OUTPUT_DIR}"
fi

# ---------------------------------------------------------------------------
# Run figure generation
# ---------------------------------------------------------------------------
conda run -p "${ENV_PREFIX}" --no-capture-output python \
    scripts/plot_defense_figures.py \
    --experiments-root "${EXPERIMENTS_ROOT}" \
    --exp-prefix       "${EXP_PREFIX}"       \
    --primary-params   mu                    \
    --tseg             1000                  \
    --w2-tsegs         1000 3000             \
    ${EXTRA_ARGS}

echo "=================================================="
echo "Defense figures complete."
echo "=================================================="
