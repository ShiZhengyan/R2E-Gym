#!/bin/bash

# Check if predictions_path argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <predictions_path>"
    echo "Example: $0 traj/r2egym-32b-agent-swebench-eval-20250602_153158.predictions.json"
    exit 1
fi

python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Verified \
    --predictions_path "$1" \
    --max_workers 24 \
    --run_id swebench_verified
