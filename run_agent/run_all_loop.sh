#!/bin/bash
# filepath: run_swe_agent_lm_32b.sh

# Exit on any error
set -e

# Check for required arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_name> <api_endpoint> [max_workers] [k] [start_idx]"
    echo ""
    echo "Arguments:"
    echo "  model_name    : Model name (e.g., SWE-bench/SWE-agent-LM-32B)"
    echo "                  'openai/' prefix will be added automatically"
    echo "  api_endpoint  : API endpoint URL (e.g., http://localhost:8007)"
    echo "  max_workers   : Number of workers (default: 24)"
    echo "  k             : Number of test cases (default: 500)"
    echo "  start_idx     : Starting index (default: 0)"
    echo ""
    echo "Example:"
    echo "  $0 SWE-bench/SWE-agent-LM-32B http://localhost:8007"
    echo "  $0 SWE-bench/SWE-agent-LM-32B http://localhost:8007 24 100 0"
    exit 1
fi

echo "Starting SWE-Bench evaluation with SWE-agent..."

# Parse command-line arguments
RAW_MODEL_NAME="$1"
API_ENDPOINT="$2"
MAX_WORKERS="${3:-8}"  # Default to 24 if not provided
K="${4:-500}"           # Default to 500 if not provided
START_IDX="${5:-0}"     # Default to 0 if not provided

# Configuration
MODEL_NAME="openai/${RAW_MODEL_NAME}"
DATASET="R2E-Gym/SWE-Bench-Verified"
SPLIT="test"
MAX_STEPS=75
TEMPERATURE=0.5
MODEL_BASENAME=$(basename "${MODEL_NAME}")
DATASET_BASENAME=$(basename "${DATASET}")
BASE_EXP_NAME=$(echo "${MODEL_BASENAME}-${DATASET_BASENAME}" | tr '[:upper:]' '[:lower:]')
TRAJ_DIR="./traj"
export OPENAI_API_KEY="not-needed"

# Create trajectory directory if it doesn't exist
mkdir -p "${TRAJ_DIR}"

echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  API Endpoint: ${API_ENDPOINT}"
echo "  Dataset: ${DATASET}"
echo "  Split: ${SPLIT}"
echo "  Max Workers: ${MAX_WORKERS}"
echo "  Test Cases: ${K}"
echo "  Max Steps: ${MAX_STEPS}"
echo "  Base Experiment Name: ${BASE_EXP_NAME}"
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

# Run 3 iterations
for ITERATION in {1..3}; do
    echo "=========================================="
    echo "Starting Iteration ${ITERATION}/3"
    echo "=========================================="
    
    # Update experiment name with iteration
    EXP_NAME="${BASE_EXP_NAME}-iter${ITERATION}"
    mkdir -p "run_logs/${EXP_NAME}"
    
    echo "Iteration ${ITERATION} Configuration:"
    echo "  Experiment Name: ${EXP_NAME}"
    echo ""

    # Step 1: Run the trajectory generation
    echo "Step 1: Generating trajectories (Iteration ${ITERATION})..."
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
      --use_existing False

    if [ $? -ne 0 ]; then
        echo "Error: Trajectory generation failed for iteration ${ITERATION}"
        exit 1
    fi

    # Step 2: Find the generated trajectory file
    echo "Step 2: Finding generated trajectory file (Iteration ${ITERATION})..."
    TRAJECTORY_FILE=$(find "${TRAJ_DIR}" -name "*.jsonl" -type f | grep "${EXP_NAME}" | sort | tail -1)

    if [ -z "$TRAJECTORY_FILE" ]; then
        echo "Error: No trajectory file found in ${TRAJ_DIR} for iteration ${ITERATION}"
        echo "Looking for files matching pattern: *${EXP_NAME}*.jsonl"
        ls -la "${TRAJ_DIR}"/*.jsonl 2>/dev/null || echo "No .jsonl files found"
        exit 1
    fi

    echo "Found trajectory file: ${TRAJECTORY_FILE}"

    # Step 3: Convert to SWE-bench format
    echo "Step 3: Converting to SWE-bench format (Iteration ${ITERATION})..."
    python run_agent/convert_r2egym_to_swebench.py "${TRAJECTORY_FILE}"

    if [ $? -ne 0 ]; then
        echo "Error: Conversion to SWE-bench format failed for iteration ${ITERATION}"
        exit 1
    fi

    # Step 4: Find the predictions file
    PREDICTIONS_FILE="${TRAJECTORY_FILE%.jsonl}.predictions.json"

    if [ ! -f "$PREDICTIONS_FILE" ]; then
        echo "Error: Predictions file not found: ${PREDICTIONS_FILE}"
        exit 1
    fi

    echo "Generated predictions file: ${PREDICTIONS_FILE}"

    # Step 5: Run SWE-bench evaluation
    echo "Step 4: Running SWE-bench evaluation (Iteration ${ITERATION})..."
    python -m swebench.harness.run_evaluation \
        --dataset_name SWE-bench/SWE-bench_Verified \
        --predictions_path "${PREDICTIONS_FILE}" \
        --max_workers ${MAX_WORKERS} \
        --run_id "swebench_verified_${EXP_NAME}" \
        --report_dir "swebench_eval_results_${EXP_NAME}"

    if [ $? -eq 0 ]; then
        echo ""
        echo "Iteration ${ITERATION} completed successfully!"
        echo "Trajectory file: ${TRAJECTORY_FILE}"
        echo "Predictions file: ${PREDICTIONS_FILE}"
        echo "Evaluation results: swebench_eval_results_${EXP_NAME}"
        echo ""
    else
        echo "Error: SWE-bench evaluation failed for iteration ${ITERATION}"
        exit 1
    fi
done

echo "=========================================="
echo "All 3 iterations completed successfully!"
echo "=========================================="
