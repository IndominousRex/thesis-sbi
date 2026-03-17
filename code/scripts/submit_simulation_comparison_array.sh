#!/bin/bash -l
# ==============================================================================
# Submit helper for comparison job arrays.
#
# This script submits:
#   1. An optional shared-dataset prep job.
#   2. One SLURM job array where each array element runs one method on one GPU.
#
# Run this helper directly with bash.
# ==============================================================================

set -euo pipefail

NUM_SIMULATIONS=${1:-2000}
T_SEG=${2:-1000}
EXP_NAME=${3:-simformer_compare}
MODE=${4:-no}
METHODS=${5:-"npe npse simformer"}
SEED=${6:-42}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SUBMIT_DIR=$(pwd)
LOG_DIR="${SUBMIT_DIR}/slurm_logs/${EXP_NAME}"
mkdir -p "${LOG_DIR}"

read -r -a METHOD_ARRAY <<< "${METHODS}"
NUM_METHODS=${#METHOD_ARRAY[@]}
if [ "${NUM_METHODS}" -eq 0 ]; then
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

ARRAY_RANGE="0-$((NUM_METHODS - 1))%${NUM_METHODS}"
JOB_MAP_FILE="${LOG_DIR}/submitted_jobs.txt"
: > "${JOB_MAP_FILE}"

echo "=================================================="
echo "Comparison Array Submission"
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
echo "Array Range:     ${ARRAY_RANGE}"
echo "Shared Methods:  ${shared_method_count}"
echo "Log Dir:         ${LOG_DIR}"
echo "=================================================="

prep_job_id=""
if [ "${shared_method_count}" -gt 1 ]; then
    prep_job_id=$(sbatch --parsable \
        --output "${LOG_DIR}/prep_%j.out" \
        --error "${LOG_DIR}/prep_%j.err" \
        "${SCRIPT_DIR}/run_prepare_shared_dataset.sh" \
        "${NUM_SIMULATIONS}" "${T_SEG}" "${EXP_NAME}" "${MODE}" "${SEED}")
    prep_job_id=${prep_job_id%%;*}
    printf "%s\t%s\t%s\n" "prep" "${prep_job_id}" "none" >> "${JOB_MAP_FILE}"
    echo "Submitted shared-dataset prep job: ${prep_job_id}"
else
    echo "No shared-dataset prep job needed."
fi

array_submit_args=(
    --parsable
    --array "${ARRAY_RANGE}"
    --output "${LOG_DIR}/compare_%A_%a.out"
    --error "${LOG_DIR}/compare_%A_%a.err"
)

if [ -n "${prep_job_id}" ]; then
    array_submit_args+=(--dependency "afterok:${prep_job_id}")
fi

array_job_id=$(sbatch "${array_submit_args[@]}" \
    "${SCRIPT_DIR}/run_simulation_comparison.sh" \
    "${NUM_SIMULATIONS}" "${T_SEG}" "${EXP_NAME}" "${MODE}" "${METHODS}" "${SEED}")
array_job_id=${array_job_id%%;*}

printf "%s\t%s\t%s\n" "array" "${array_job_id}" "${prep_job_id:-none}" >> "${JOB_MAP_FILE}"

echo "Submitted comparison array job: ${array_job_id}"
echo "=================================================="
echo "Submitted jobs listed in: ${JOB_MAP_FILE}"
echo "Array workers will not write comparison summary JSON files."
echo "Use the shared run-group suffix A<array_job_id> to identify related outputs."
echo "=================================================="
