#!/bin/bash -l
# ==============================================================================
# SLURM submission helper for comparison experiments.
#
# This launcher submits one separate 1-GPU SLURM job per method. If more than
# one shared-data method is requested (NPE/NPSE/Simformer), it first submits a
# lightweight dataset-preparation job so the method jobs can all reuse the same
# cached dataset without racing to create it.
#
# Run this script directly with `bash`, not `sbatch`.
#
# Usage:
#   bash scripts/run_simulation_comparison.sh
#   bash scripts/run_simulation_comparison.sh 20000 1000 cmp_full no "npe npse fnpe simformer" 42
#   bash scripts/run_simulation_comparison.sh 64 128 cmp_smoke smoke "npe npse fnpe simformer" 42
#
# Positional args:
#   $1  NUM_SIMULATIONS  (default: 2000)
#   $2  T_SEG            (default: 1000)
#   $3  EXP_NAME         (default: simformer_compare)
#   $4  MODE             (default: no)   one of: no | yes | smoke
#   $5  METHODS          (default: "npe npse simformer")  space-separated in quotes
#   $6  SEED             (default: 42)
#
# Environment overrides:
#   GPU_PARTITION        Optional SLURM partition for GPU child jobs
#   GPU_GRES             GPU request for child jobs (default: gpu:1)
#   GPU_CPUS_PER_TASK    CPUs per GPU child job (default: 8)
#   GPU_MEM_PER_CPU      Memory per CPU for GPU child jobs (default: 8G)
#   GPU_TIME             Walltime for GPU child jobs (default: 24:00:00)
#   PREP_PARTITION       Optional partition for the CPU dataset-prep job
#   PREP_CPUS_PER_TASK   CPUs for prep job (default: 8)
#   PREP_MEM_PER_CPU     Memory per CPU for prep job (default: 8G)
#   PREP_TIME            Walltime for prep job (default: 02:00:00)
# ==============================================================================

set -euo pipefail

NUM_SIMULATIONS=${1:-2000}
T_SEG=${2:-1000}
EXP_NAME=${3:-simformer_compare}
MODE=${4:-no}
METHODS=${5:-"npe npse simformer"}
SEED=${6:-42}

GPU_PARTITION=${GPU_PARTITION:-}
GPU_GRES=${GPU_GRES:-gpu:1}
GPU_CPUS_PER_TASK=${GPU_CPUS_PER_TASK:-8}
GPU_MEM_PER_CPU=${GPU_MEM_PER_CPU:-8G}
GPU_TIME=${GPU_TIME:-24:00:00}

PREP_PARTITION=${PREP_PARTITION:-}
PREP_CPUS_PER_TASK=${PREP_CPUS_PER_TASK:-8}
PREP_MEM_PER_CPU=${PREP_MEM_PER_CPU:-8G}
PREP_TIME=${PREP_TIME:-02:00:00}

MODE_FLAG=""
if [ "${MODE}" = "yes" ]; then
    MODE_FLAG="--quick"
elif [ "${MODE}" = "smoke" ]; then
    MODE_FLAG="--smoke"
elif [ "${MODE}" != "no" ]; then
    echo "Unsupported mode: ${MODE}"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
LOG_DIR="${SUBMIT_DIR}/slurm_logs/${EXP_NAME}"
mkdir -p "${LOG_DIR}"

read -r -a METHOD_ARRAY <<< "${METHODS}"
if [ ${#METHOD_ARRAY[@]} -eq 0 ]; then
    echo "No methods specified."
    exit 1
fi

shared_method_count=0
for method in "${METHOD_ARRAY[@]}"; do
    case "${method}" in
        npe|npse|fnpe|simformer)
            ;;
        *)
            echo "Unsupported method: ${method}"
            exit 1
            ;;
    esac

    case "${method}" in
        npe|npse|simformer)
            shared_method_count=$((shared_method_count + 1))
            ;;
    esac
done

echo "=================================================="
echo "Comparison Submission"
echo "=================================================="
echo "Date:            $(date)"
echo "Submit Dir:      ${SUBMIT_DIR}"
echo "Code Dir:        ${CODE_DIR}"
echo "Num Simulations: ${NUM_SIMULATIONS}"
echo "T_seg:           ${T_SEG}"
echo "Exp Name:        ${EXP_NAME}"
echo "Mode:            ${MODE}"
echo "Methods:         ${METHODS}"
echo "Seed:            ${SEED}"
echo "Shared Methods:  ${shared_method_count}"
echo "Log Dir:         ${LOG_DIR}"
echo "=================================================="

submit_prep_job() {
    local wrap_cmd
    wrap_cmd="bash -lc 'cd ${CODE_DIR} && module load Miniforge3 && conda activate /software/NHKB22930/nhkbarit/conda_envs/npe && export XLA_PYTHON_CLIENT_PREALLOCATE=false && export JAX_PLATFORM_NAME=cpu && python scripts/prepare_shared_dataset.py --exp-name ${EXP_NAME} --num-simulations ${NUM_SIMULATIONS} --T-seg ${T_SEG} --device cpu --seed ${SEED} ${MODE_FLAG}'"

    local sbatch_args=(
        --parsable
        --job-name "${EXP_NAME}_prep"
        --nodes 1
        --ntasks 1
        --cpus-per-task "${PREP_CPUS_PER_TASK}"
        --mem-per-cpu "${PREP_MEM_PER_CPU}"
        --time "${PREP_TIME}"
        --output "${LOG_DIR}/prep_%j.out"
        --error "${LOG_DIR}/prep_%j.err"
    )

    if [ -n "${PREP_PARTITION}" ]; then
        sbatch_args+=(--partition "${PREP_PARTITION}")
    fi

    sbatch_args+=(--wrap "${wrap_cmd}")
    local job_id
    job_id=$(sbatch "${sbatch_args[@]}")
    echo "${job_id%%;*}"
}

submit_method_job() {
    local method="$1"
    local dependency_job="${2:-}"
    local child_exp_name="${EXP_NAME}_${method}"
    local wrap_cmd

    wrap_cmd="bash -lc 'cd ${CODE_DIR} && module load Miniforge3 && conda activate /software/NHKB22930/nhkbarit/conda_envs/npe && export XLA_PYTHON_CLIENT_PREALLOCATE=false && export JAX_PLATFORM_NAME=cpu && python scripts/run_simulation_comparison.py --exp-name ${child_exp_name} --num-simulations ${NUM_SIMULATIONS} --T-seg ${T_SEG} --device cuda --methods ${method} --seed ${SEED} ${MODE_FLAG}'"

    local sbatch_args=(
        --parsable
        --job-name "${EXP_NAME}_${method}"
        --nodes 1
        --ntasks 1
        --cpus-per-task "${GPU_CPUS_PER_TASK}"
        --gres "${GPU_GRES}"
        --mem-per-cpu "${GPU_MEM_PER_CPU}"
        --time "${GPU_TIME}"
        --output "${LOG_DIR}/${method}_%j.out"
        --error "${LOG_DIR}/${method}_%j.err"
    )

    if [ -n "${GPU_PARTITION}" ]; then
        sbatch_args+=(--partition "${GPU_PARTITION}")
    fi

    if [ -n "${dependency_job}" ]; then
        sbatch_args+=(--dependency "afterok:${dependency_job}")
    fi

    sbatch_args+=(--wrap "${wrap_cmd}")
    local job_id
    job_id=$(sbatch "${sbatch_args[@]}")
    echo "${job_id%%;*}"
}

JOB_MAP_FILE="${LOG_DIR}/submitted_jobs.txt"
: > "${JOB_MAP_FILE}"

prep_job_id=""
if [ "${shared_method_count}" -gt 1 ]; then
    prep_job_id=$(submit_prep_job)
    echo "Submitted shared-dataset prep job: ${prep_job_id}"
    printf "%s\t%s\t%s\n" "prep" "${prep_job_id}" "none" >> "${JOB_MAP_FILE}"
else
    echo "No shared-dataset prep job needed."
fi

for method in "${METHOD_ARRAY[@]}"; do
    dependency=""
    case "${method}" in
        npe|npse|simformer)
            dependency="${prep_job_id}"
            ;;
    esac

    job_id=$(submit_method_job "${method}" "${dependency}")
    printf "%s\t%s\t%s\n" "${method}" "${job_id}" "${dependency:-none}" >> "${JOB_MAP_FILE}"

    if [ -n "${dependency}" ]; then
        echo "Submitted ${method} job ${job_id} (afterok:${dependency})"
    else
        echo "Submitted ${method} job ${job_id}"
    fi
done

echo "=================================================="
echo "Submitted jobs listed in: ${JOB_MAP_FILE}"
echo "No combined comparison summary is generated by this launcher."
echo "Each child job writes its own outputs under experiments/ and SLURM logs under ${LOG_DIR}."
echo "=================================================="
