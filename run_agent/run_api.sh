#!/bin/bash

# Exit on any error
set -e

echo "Starting SWE-Bench evaluation with R2EGym-32B-Agent..."

# Configuration
MODEL_NAME="claude-sonnet-4"
DATASET="R2E-Gym/R2E-Gym-Subset"
SPLIT="train"
MAX_WORKERS=72  # Adjust based on your system capacity
K=5000  # Number of test cases to evaluate
START_IDX=0
MAX_STEPS=50
TEMPERATURE=1.0
API_ENDPOINT="not-needed"  # vLLM server endpoint
EXP_NAME="${DATASET}-${MODEL_NAME}-${SPLIT}-${MAX_STEPS}-${TEMPERATURE}"
TRAJ_DIR="./traj"
export OPENAI_API_KEY="not-needed"

# Create trajectory directory if it doesn't exist
mkdir -p "${TRAJ_DIR}"
mkdir -p "run_logs/${EXP_NAME}"

echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  API Endpoint: ${API_ENDPOINT}"
echo "  Dataset: ${DATASET}"
echo "  Split: ${SPLIT}"
echo "  Max Workers: ${MAX_WORKERS}"
echo "  Test Cases: ${K}"
echo "  Max Steps: ${MAX_STEPS}"
echo "  Experiment Name: ${EXP_NAME}"
echo "  Output Directory: ${TRAJ_DIR}"
echo ""

# Check if vLLM server is running
echo "Checking vLLM server availability..."
if ! curl -s "${API_ENDPOINT}/v1/models" > /dev/null; then
    echo "Error: vLLM server is not running on ${API_ENDPOINT}"
    echo "Please start your vLLM server first."
    exit 1
fi

echo "vLLM server is available. Starting evaluation..."
echo ""

source .venv/bin/activate

# Run the evaluation
uv run python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --k "${K}" \
  --traj_dir "${TRAJ_DIR}" \
  --exp_name "${EXP_NAME}" \
  --start_idx "${START_IDX}" \
  --max_steps "${MAX_STEPS}" \
  --max_workers "${MAX_WORKERS}" \
  --llm_name "${MODEL_NAME}" \
  --use_fn_calling True \
  --temperature "${TEMPERATURE}" \
  --backend docker \
  --use_existing True

echo ""
echo "Evaluation completed!"
