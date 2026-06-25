#!/bin/bash
set -euo pipefail

PROJECT_DIR="/Users/rezsol/Desktop/projects/agentic_projects/research_efficiency"
TAU_BENCH="/Users/rezsol/Desktop/projects/agentic_projects/tau-bench"
export PYTHONPATH="src:${TAU_BENCH}"

cd "$PROJECT_DIR"

ABLATION_CONFIGS="ablation-no-staleness,ablation-staleness-only,ablation-no-behavioral,ablation-no-progressive,ablation-trace-context"

RUNS=(
    "retail 115 ablation_retail"
    "airline 50 ablation_airline"
)

TOTAL=${#RUNS[@]}
PASSED=0
FAILED=0
FAILED_RUNS=""

echo "========================================"
echo "Ablation Study Runs"
echo "========================================"
echo "Started: $(date)"
echo "Configs: ${ABLATION_CONFIGS}"
echo "Total runs: ${TOTAL}"
echo ""

for i in "${!RUNS[@]}"; do
    IFS=' ' read -r domain num_tasks output_dir <<< "${RUNS[$i]}"
    run_num=$((i + 1))

    echo "----------------------------------------"
    echo "Run ${run_num}/${TOTAL}: ${domain} (${num_tasks} tasks) → results/${output_dir}"
    echo "Started: $(date)"
    echo "----------------------------------------"

    if python -m adaptive_compression experiment \
        --domain "$domain" \
        --num-tasks "$num_tasks" \
        --provider deepseek \
        --model deepseek-v4-flash \
        --output "results/${output_dir}" \
        --configs "$ABLATION_CONFIGS"; then
        PASSED=$((PASSED + 1))
        echo ""
        echo "Run ${run_num} COMPLETED at $(date)"
        echo ""
    else
        exit_code=$?
        FAILED=$((FAILED + 1))
        FAILED_RUNS="${FAILED_RUNS}  - ${output_dir} (exit code ${exit_code})\n"
        echo ""
        echo "Run ${run_num} FAILED (exit code ${exit_code}) at $(date)"
        echo "Continuing to next run..."
        echo ""
    fi

    if [ "$run_num" -lt "$TOTAL" ]; then
        echo "Pausing 30s between runs..."
        sleep 30
    fi
done

echo "========================================"
echo "ALL ABLATION RUNS COMPLETE"
echo "========================================"
echo "Finished: $(date)"
echo "Passed: ${PASSED}/${TOTAL}"
echo "Failed: ${FAILED}/${TOTAL}"
if [ -n "$FAILED_RUNS" ]; then
    echo ""
    echo "Failed runs:"
    echo -e "$FAILED_RUNS"
fi
