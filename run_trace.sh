#!/bin/bash
set -euo pipefail

PROJECT_DIR="/Users/rezsol/Desktop/projects/agentic_projects/research_efficiency"
TAU_BENCH="/Users/rezsol/Desktop/projects/agentic_projects/tau-bench"

cd "$PROJECT_DIR"

# Load API keys from .env
set -a
source .env
set +a

export PYTHONPATH="src:${TAU_BENCH}"

# Which trace variant to run. Override on the command line, e.g.:
#   ./run_trace.sh ablation-trace-context
CONFIG="${1:-ablation-trace-compact}"
DOMAIN="retail"
NUM_TASKS=50
CONCURRENCY=4
OUTPUT_DIR="results/${CONFIG//ablation-/}_retail50"

echo "========================================"
echo "Trace Variant Run (no baseline)"
echo "========================================"
echo "Started:     $(date)"
echo "Config:      ${CONFIG}"
echo "Domain:      ${DOMAIN}"
echo "Tasks:       ${NUM_TASKS}"
echo "Concurrency: ${CONCURRENCY}"
echo "Output:      ${OUTPUT_DIR}"
echo ""

python -m adaptive_compression experiment \
    --domain "$DOMAIN" \
    --num-tasks "$NUM_TASKS" \
    --provider deepseek \
    --model deepseek-v4-flash \
    --concurrency "$CONCURRENCY" \
    --output "$OUTPUT_DIR" \
    --configs "$CONFIG"

echo ""
echo "========================================"
echo "DONE: $(date)"
echo "Results: ${OUTPUT_DIR}/"
echo "========================================"
