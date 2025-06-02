#!/bin/bash
# filepath: run_qwen_32b_quick_test.sh

# Exit on any error
set -e

# Configuration
MODEL_NAME="openai/Qwen/Qwen2.5-Coder-32B-Instruct"
DATASET="R2E-Gym/SWE-Bench-Verified"
SPLIT="test"
MAX_WORKERS=1  # Reduced for quick test
K=1  # Only 10 test cases for quick validation
START_IDX=0
MAX_STEPS=40
TEMPERATURE=0
API_ENDPOINT="http://localhost:8002"  # vLLM server endpoint
EXP_NAME="qwen2.5-coder-32b-swebench-quicktest-$(date +%Y%m%d_%H%M%S)"
TRAJ_DIR="./traj"
export OPENAI_API_KEY="not-needed"

# Create trajectory directory if it doesn't exist
mkdir -p "${TRAJ_DIR}"
mkdir -p "run_logs/${EXP_NAME}"

echo "Starting SWE-Bench quick test with (${K} examples)..."
echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  API Endpoint: ${API_ENDPOINT}"
echo "  Dataset: ${DATASET}"
echo "  Split: ${SPLIT}"
echo "  Max Workers: ${MAX_WORKERS}"
echo "  Test Cases: ${K} (QUICK TEST)"
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

echo "vLLM server is available. Starting quick test..."
echo ""

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
echo "Quick test completed! Check results in ${TRAJ_DIR}/${EXP_NAME}.jsonl"
echo "This was a quick validation with only ${K} test cases."
