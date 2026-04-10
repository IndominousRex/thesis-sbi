#!/bin/bash -l
# ==============================================================================
# Slurm CPU job to run the aggregate_simulation_benchmark.py script.
#
# Usage (from the code/ directory):
#   sbatch scripts/run_aggregate_benchmark.sh <exp_prefix> [experiments_root]
#
# Examples:
#   sbatch scripts/run_aggregate_benchmark.sh bench_budget_v3_fixed
#   sbatch scripts/run_aggregate_benchmark.sh bench_budget_v3_fixed experiments
# ==============================================================================

#SBATCH --job-name=agg_bench
#SBATCH --partition=small_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=agg_bench_%j.out
#SBATCH --error=agg_bench_%j.err

set -euo pipefail

EXP_PREFIX=${1:?exp_prefix argument required}
EXPERIMENTS_ROOT=${2:-experiments}

# ---------------------------------------------------------------------------
# Locate code root (supports both interactive and sbatch invocation)
# ---------------------------------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/aggregate_simulation_benchmark.py" ]; then
    CODE_DIR="${SLURM_SUBMIT_DIR}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/aggregate_simulation_benchmark.py" ]; then
    CODE_DIR=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)
elif [ -d "/bigwork/nhkbarit/thesis-code/code" ]; then
    CODE_DIR="/bigwork/nhkbarit/thesis-code/code"
else
    echo "Could not determine code directory."
    echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
    exit 1
fi

echo "=================================================="
echo "Aggregate Benchmark"
echo "=================================================="
echo "CODE_DIR:         ${CODE_DIR}"
echo "EXP_PREFIX:       ${EXP_PREFIX}"
echo "EXPERIMENTS_ROOT: ${EXPERIMENTS_ROOT}"
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
# Run aggregation
# ---------------------------------------------------------------------------
conda run -p "${ENV_PREFIX}" --no-capture-output python \
    scripts/aggregate_simulation_benchmark.py \
    --experiments-root "${EXPERIMENTS_ROOT}" \
    --exp-prefix "${EXP_PREFIX}"

echo "=================================================="
echo "Aggregation complete."
echo "=================================================="
