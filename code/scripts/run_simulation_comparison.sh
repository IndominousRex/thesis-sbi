#!/bin/bash -l
# ==============================================================================
# SLURM script for Simformer comparison experiments
#
# Runs a multi-method comparison. NPE, NPSE, and Simformer share the same
# cached dataset; FNPE keeps its own task-specific simulation path.
#
# Usage:
#   sbatch run_simulation_comparison.sh                              # defaults
#   sbatch run_simulation_comparison.sh 5000 1000 sim_cmp            # custom sims, T_seg, name
#   sbatch run_simulation_comparison.sh 500 1000 quick_test yes      # quick mode
#   sbatch run_simulation_comparison.sh 20000 3000 cmp_smoke smoke   # minimal smoke test
#   sbatch --gres=gpu:4 run_simulation_comparison.sh ...             # parallel model runs on 4 visible GPUs
#
# Positional args:
#   $1  NUM_SIMULATIONS  (default: 2000)
#   $2  T_SEG            (default: 1000)
#   $3  EXP_NAME         (default: simformer_compare)
#   $4  MODE             (default: no)   one of: no | yes | smoke
#   $5  METHODS          (default: "npe npse simformer")  space-separated in quotes
#   $6  SEED             (default: 42)
# ==============================================================================

#SBATCH --job-name=simformer_cmp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=simformer_cmp_%j.out
#SBATCH --error=simformer_cmp_%j.err

# ==============================================================================
# Parse arguments
# ==============================================================================
NUM_SIMULATIONS=${1:-2000}
T_SEG=${2:-1000}
EXP_NAME=${3:-simformer_compare}
MODE=${4:-no}
METHODS=${5:-"npe npse simformer"}
SEED=${6:-42}

# ==============================================================================
# Build flags
# ==============================================================================
MODE_FLAG=""
if [ "${MODE}" = "yes" ]; then
    MODE_FLAG="--quick"
elif [ "${MODE}" = "smoke" ]; then
    MODE_FLAG="--smoke"
fi

# ==============================================================================
# Environment setup
# ==============================================================================
set -e

echo "=================================================="
echo "Simformer Comparison Experiment"
echo "=================================================="
echo "Date:            $(date)"
echo "Node:            $(hostname)"
echo "Job ID:          ${SLURM_JOB_ID:-local}"
echo "Num Simulations: ${NUM_SIMULATIONS}"
echo "T_seg:           ${T_SEG}"
echo "Exp Name:        ${EXP_NAME}"
echo "Mode:            ${MODE}"
echo "Methods:         ${METHODS}"
echo "Seed:            ${SEED}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "=================================================="

# --- Go to code directory ---
cd /bigwork/nhkbarit/thesis-code/code

# --- Load environment ---
module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

# --- JAX configuration (use CPU for JAX, GPU for PyTorch) ---
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=cpu

# ==============================================================================
# Run comparison
# ==============================================================================
# Parallel model execution is automatic when multiple GPUs are visible to the job.
srun python scripts/run_simulation_comparison.py \
    --exp-name "${EXP_NAME}" \
    --num-simulations ${NUM_SIMULATIONS} \
    --T-seg ${T_SEG} \
    --device cuda \
    --methods ${METHODS} \
    --seed ${SEED} \
    ${MODE_FLAG}

echo "=================================================="
echo "[$(date)] Comparison completed"
echo "=================================================="
