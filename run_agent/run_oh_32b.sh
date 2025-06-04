#!/bin/bash

# Exit on any error
set -e

echo "Starting SWE-Bench evaluation with OpenHands-32B-Agent_qwen32B_bs1x8_lr1e-4_ep5..."

# Configuration
MODEL_NAME="openai/OpenHands-32B-Agent_qwen32B_bs1x8_lr1e-4_ep5"
DATASET="R2E-Gym/SWE-Bench-Verified"
SPLIT="test"
MAX_WORKERS=2  # Adjust based on your system capacity
K=500  # Number of test cases to evaluate
START_IDX=0
MAX_STEPS=40
TEMPERATURE=0
API_ENDPOINT="http://localhost:8010"  # vLLM server endpoint
EXP_NAME="OpenHands-32B-Agent_qwen32B_bs1x8_lr1e-4_ep5-swebench-eval-$(date +%Y%m%d_%H%M%S)"
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
  --llm_base_url "${API_ENDPOINT}/v1" \
  --use_fn_calling False \
  --temperature "${TEMPERATURE}" \
  --use_existing True

echo ""
echo "Evaluation completed!"
