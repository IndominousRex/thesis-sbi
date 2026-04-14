#!/bin/bash -l
# ==============================================================================
# SLURM array job: real-data evaluation for all 12 method×param-set combinations
#
# Usage:
#   sbatch --array=0-11 run_real_data_eval.sh
#
# Each array task loads a trained checkpoint and runs real-data inference
# (PPC on Jeversen_2022_10_12_110132.csv + multi-trajectory PPC).
# ==============================================================================

#SBATCH --job-name=real_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=real_eval_%A_%a.out
#SBATCH --error=real_eval_%A_%a.err

# ==============================================================================
# Job definitions: METHOD  CHECKPOINT_DIR
# ==============================================================================
METHODS=(
  npe
  npe
  npe
  npse
  npse
  npse
  fnpe
  fnpe
  fnpe
  simformer
  simformer
  simformer
)

CHECKPOINTS=(
  "bench_budget_v3_fixed_b20000000_pmu_t1000_s43_npe_pmu_t1000_s43_tr44_b20000000_20260406-013639"
  "bench_budget_v3_fixed_b40000000_pmu_cd_t1000_s43_npe_pmu_cd_t1000_s43_tr44_b40000000_20260406-215012"
  "bench_budget_v3_fixed_b20000000_pmu_cd_m_t1000_s44_npe_pmu_cd_m_t1000_s44_tr45_b20000000_20260412-010920"
  "bench_budget_v3_fixed_b60000000_pmu_t1000_s46_npse_pmu_t1000_s46_tr47_b60000000_20260411-121143"
  "bench_budget_v3_fixed_b60000000_pmu_cd_t1000_s42_npse_pmu_cd_t1000_s42_tr43_b60000000_20260405-192917"
  "bench_budget_v3_fixed_b60000000_pmu_cd_m_t2000_s42_npse_pmu_cd_m_t2000_s42_tr43_b60000000_20260405-214553"
  "bench_budget_v3_fixed_b60000000_pmu_t2000_s46_fnpe_pmu_t2000_s46_tr47_b60000000_20260410-232008"
  "bench_budget_v3_fixed_b20000000_pmu_cd_t1000_s42_fnpe_pmu_cd_t1000_s42_tr43_b20000000_20260404-135124"
  "bench_budget_v3_fixed_b20000000_pmu_cd_m_t1000_s44_fnpe_pmu_cd_m_t1000_s44_tr45_b20000000_20260409-222417"
  "bench_budget_v3_fixed_b40000000_pmu_t1000_s44_simformer_pmu_t1000_s44_tr45_b40000000_20260411-063031"
  "bench_budget_v3_fixed_b60000000_pmu_cd_t1000_s46_simformer_pmu_cd_t1000_s46_tr47_b60000000_20260411-201343"
  "bench_budget_v3_fixed_b60000000_pmu_cd_m_t2000_s43_simformer_pmu_cd_m_t2000_s43_tr44_b60000000_20260408-221851"
)

LABELS=(
  "npe_mu"
  "npe_mu_cd"
  "npe_mu_cd_m"
  "npse_mu"
  "npse_mu_cd"
  "npse_mu_cd_m"
  "fnpe_mu"
  "fnpe_mu_cd"
  "fnpe_mu_cd_m"
  "simformer_mu"
  "simformer_mu_cd"
  "simformer_mu_cd_m"
)

# ==============================================================================
# Pick this task's configuration
# ==============================================================================
IDX=${SLURM_ARRAY_TASK_ID}
METHOD=${METHODS[$IDX]}
CKPT=${CHECKPOINTS[$IDX]}
LABEL=${LABELS[$IDX]}

REAL_CSV="../data/measurements/Jeversen_2022_10_12_110132.csv"
REAL_DIR="../data/measurements"
CKPT_BASE="/bigwork/nhkbarit/thesis-code/code/experiments"

echo "=================================================="
echo "Real-Data Evaluation: ${LABEL}"
echo "=================================================="
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID} / Task ${IDX}"
echo "Method:     ${METHOD}"
echo "Checkpoint: ${CKPT}"
echo "=================================================="

# ==============================================================================
# Environment setup
# ==============================================================================
set -e

cd /bigwork/nhkbarit/thesis-code/code

module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORM_NAME || true

# ==============================================================================
# Run evaluation
# ==============================================================================
srun python run.py \
    --method ${METHOD} \
    --exp-name real_eval_${LABEL} \
    --eval \
    --checkpoint "${CKPT_BASE}/${CKPT}" \
    --real-data-csv "${REAL_CSV}" \
    --real-data-dir "${REAL_DIR}" \
    --k-ppc 300 \
    --device cuda

echo "=================================================="
echo "[$(date)] Completed: ${LABEL}"
echo "=================================================="
