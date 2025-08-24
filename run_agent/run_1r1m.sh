#!/bin/bash

# Exit on any error
set -e

echo "Starting SWE-Bench evaluation with R2EGym-32B-Agent..."

# Configuration
MODEL_NAME="capi-claude-sonnet-4"
# MODEL_NAME="capi-o3-2025-04-16"
DATASET="/home/zhengyanshi/project/SWE-smith/logs/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/astropy__astropy.26d14786/task_insts/astropy__astropy.26d14786_ps.json"
# DATASET_SLUG=$(echo "$DATASET" | sed 's|/|--|g')
K=2313
K=10
DATASET_SLUG=astropy__astropy.26d14786_ps
SPLIT="train"
MAX_WORKERS=64
START_IDX=0
MAX_STEPS=40
MAX_STEPS_ABS=50
TEMPERATURE=1.0
K_RESPONSES=5
# BACKEND="kubernetes"
BACKEND="docker"
API_ENDPOINT="not-needed"  # vLLM server endpoint
EXP_NAME="test_${DATASET_SLUG}-${MODEL_NAME}-${SPLIT}-s${MAX_STEPS}-${MAX_STEPS_ABS}-t${TEMPERATURE}-k${K_RESPONSES}-${BACKEND}-${MAX_WORKERS}"
# EXP_NAME="${DATASET_SLUG}-${MODEL_NAME}-${SPLIT}-s${MAX_STEPS}-${MAX_STEPS_ABS}-t${TEMPERATURE}-k${K_RESPONSES}-${BACKEND}-${MAX_WORKERS}"
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

# Skip vLLM server check
echo "Skipping vLLM server check..."
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
  --max_steps_absolute "${MAX_STEPS_ABS}" \
  --max_workers "${MAX_WORKERS}" \
  --llm_name "${MODEL_NAME}" \
  --k_responses "${K_RESPONSES}" \
  --use_fn_calling True \
  --temperature "${TEMPERATURE}" \
  --backend "${BACKEND}" \
  --use_existing False \
  --use_1r1m true

echo ""
echo "Evaluation completed!"
