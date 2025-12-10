#!/bin/bash

# Evaluation Pipeline Script
# Evaluates models with different combinations of parameters and performs t-tests
#
# Usage:
#   bash evaluate_pipeline.sh <SEED>
#
# Example:
#   bash evaluate_pipeline.sh 42

set -e

# ==================== Configuration ====================

# Check if SEED is provided
if [ $# -lt 1 ]; then
    echo "Error: SEED argument is required"
    echo "Usage: bash evaluate_pipeline.sh <SEED>"
    exit 1
fi

SEED="$1"

# Base paths
FRAMES_DIR="${FRAMES_DIR:-data/sthv2/frames_224x224}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-timbrooks/instruct-pix2pix}"
PROGRESS_MODEL_PATH="${PROGRESS_MODEL_PATH:-progress_evaluator/checkpoints/checkpoint_ep20.pth}"
PROGRESS_NUM_FRAMES="${PROGRESS_NUM_FRAMES:-20}"
PROGRESS_HIDDEN_DIM="${PROGRESS_HIDDEN_DIM:-512}"

# Sampling parameters
NUM_SAMPLES="${NUM_SAMPLES:-100}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-25}"
IMAGE_GUIDANCE_SCALE="${IMAGE_GUIDANCE_SCALE:-1.5}"
INPUT_FRAME_IDX="${INPUT_FRAME_IDX:-20}"

# Base output directory
BASE_OUTPUT_DIR="test_result/seed_${SEED}"

# ==================== Parameter Combinations ====================

# All models are trained on basic data (which includes both basic and backforth)
# But we evaluate each model on both basic and backforth validation sets
TARGET_FRAMES=(21 25)
USE_PROGRESS_OPTIONS=("true" "false")
EVAL_TASKS=("basic" "backforth")  # Evaluation datasets

# ==================== Helper Functions ====================

# Function to get model directory name
# All models are trained on basic data, so task parameter is not used
get_model_dir() {
    local target_frame="$1"
    local use_progress="$2"
    
    if [ "$target_frame" = "21" ]; then
        if [ "$use_progress" = "true" ]; then
            echo "outputs/instruct_pix2pix_224_progress"
        else
            echo "outputs/instruct_pix2pix_224"
        fi
    else
        # frame25 uses _next5 suffix
        if [ "$use_progress" = "true" ]; then
            echo "outputs/instruct_pix2pix_224_progress_next5"
        else
            echo "outputs/instruct_pix2pix_224_next5"
        fi
    fi
}

# Function to get output directory name
# Includes eval_task suffix to distinguish evaluation datasets
get_output_dir() {
    local target_frame="$1"
    local use_progress="$2"
    local eval_task="$3"
    
    local base_name
    if [ "$target_frame" = "21" ]; then
        if [ "$use_progress" = "true" ]; then
            base_name="instruct_pix2pix_224_progress"
        else
            base_name="instruct_pix2pix_224"
        fi
    else
        # frame25 uses _next5 suffix
        if [ "$use_progress" = "true" ]; then
            base_name="instruct_pix2pix_224_progress_next5"
        else
            base_name="instruct_pix2pix_224_next5"
        fi
    fi
    
    # Add eval_task suffix to distinguish evaluation datasets
    if [ "$eval_task" = "backforth" ]; then
        echo "${BASE_OUTPUT_DIR}/${base_name}_eval_backforth"
    else
        echo "${BASE_OUTPUT_DIR}/${base_name}_eval_basic"
    fi
}

# Function to get validation annotations file
get_val_annotations() {
    local task="$1"
    if [ "$task" = "backforth" ]; then
        echo "data/sthv2/annotations/val_filtered_backforth.json"
    else
        echo "data/sthv2/annotations/val_filtered.json"
    fi
}

# ==================== Main Execution ====================

echo "========================================="
echo "Evaluation Pipeline"
echo "========================================="
echo "Seed: ${SEED}"
echo "Base output directory: ${BASE_OUTPUT_DIR}"
echo ""
echo "Parameter combinations:"
echo "  Target frames: ${TARGET_FRAMES[@]}"
echo "  Use progress: ${USE_PROGRESS_OPTIONS[@]}"
echo "  Evaluation datasets: ${EVAL_TASKS[@]}"
echo "Total combinations: $(( ${#TARGET_FRAMES[@]} * ${#USE_PROGRESS_OPTIONS[@]} * ${#EVAL_TASKS[@]} ))"
echo "========================================="
echo ""

# Create base output directory
mkdir -p "${BASE_OUTPUT_DIR}"

# Track all evaluation directories for t-test
declare -A eval_dirs

# Run evaluation for each combination
combination_count=0
total_combinations=$((${#TARGET_FRAMES[@]} * ${#USE_PROGRESS_OPTIONS[@]} * ${#EVAL_TASKS[@]}))

for target_frame in "${TARGET_FRAMES[@]}"; do
    for use_progress in "${USE_PROGRESS_OPTIONS[@]}"; do
        for eval_task in "${EVAL_TASKS[@]}"; do
            combination_count=$((combination_count + 1))
            
            # Get configuration
            model_dir=$(get_model_dir "$target_frame" "$use_progress")
            output_dir=$(get_output_dir "$target_frame" "$use_progress" "$eval_task")
            val_annotations=$(get_val_annotations "$eval_task")
            
            # Create output directory
            mkdir -p "$output_dir"
            
            # Store directory for t-test
            key="frame${target_frame}_${use_progress}_eval${eval_task}"
            eval_dirs["$key"]="$output_dir"
            
            echo ""
            echo "========================================="
            echo "Combination ${combination_count}/${total_combinations}"
            echo "========================================="
            echo "Model: frame${target_frame}, progress=${use_progress}"
            echo "Evaluation dataset: $eval_task"
            echo "Model directory: $model_dir"
            echo "Output directory: $output_dir"
            echo "========================================="
            echo ""
            
            # Run evaluation
            export MODEL_DIR="$model_dir"
            export PRETRAINED_MODEL="$PRETRAINED_MODEL"
            export FRAMES_DIR="$FRAMES_DIR"
            export VAL_ANNOTATIONS="$val_annotations"
            export OUTPUT_DIR="$output_dir"
            export INPUT_FRAME_IDX="$INPUT_FRAME_IDX"
            export TARGET_FRAME_IDX="$target_frame"
            export NUM_SAMPLES="$NUM_SAMPLES"
            export SEED="$SEED"
            export NUM_INFERENCE_STEPS="$NUM_INFERENCE_STEPS"
            export IMAGE_GUIDANCE_SCALE="$IMAGE_GUIDANCE_SCALE"
            export USE_PROGRESS_ESTIMATOR="$use_progress"
            export PROGRESS_MODEL_PATH="$PROGRESS_MODEL_PATH"
            export PROGRESS_NUM_FRAMES="$PROGRESS_NUM_FRAMES"
            export PROGRESS_HIDDEN_DIM="$PROGRESS_HIDDEN_DIM"
            
            # Run evaluate.sh (suppress set -e temporarily to continue on error)
            set +e
            bash evaluate.sh
            eval_exit_code=$?
            set -e
            
            if [ $eval_exit_code -ne 0 ]; then
                echo "Warning: Evaluation failed for combination (frame=$target_frame, progress=$use_progress, eval_task=$eval_task)"
                echo "Continuing with next combination..."
            fi
        done
    done
done

echo ""
echo "========================================="
echo "All Evaluations Complete"
echo "========================================="
echo ""

# ==================== Perform T-Tests ====================

echo "========================================="
echo "Performing T-Tests"
echo "========================================="
echo ""

# Prepare t-test pairs
# Format: (target_frame, eval_task, progress_dir, no_progress_dir)
# Compare models with/without progress for each (target_frame, eval_task) combination
t_test_pairs=()

for target_frame in "${TARGET_FRAMES[@]}"; do
    for eval_task in "${EVAL_TASKS[@]}"; do
        progress_key="frame${target_frame}_true_eval${eval_task}"
        no_progress_key="frame${target_frame}_false_eval${eval_task}"
        
        progress_dir="${eval_dirs[$progress_key]}"
        no_progress_dir="${eval_dirs[$no_progress_key]}"
        
        if [ -n "$progress_dir" ] && [ -n "$no_progress_dir" ]; then
            t_test_pairs+=("$target_frame|$eval_task|$progress_dir|$no_progress_dir")
        fi
    done
done

# Run t-tests
for pair in "${t_test_pairs[@]}"; do
    IFS='|' read -r target_frame eval_task progress_dir no_progress_dir <<< "$pair"
    
    echo "Running t-test for: frame=$target_frame, eval_dataset=$eval_task"
    echo "  With progress: $progress_dir"
    echo "  Without progress: $no_progress_dir"
    
    # Check if results.json exists in both directories
    if [ ! -f "$progress_dir/results.json" ] || [ ! -f "$no_progress_dir/results.json" ]; then
        echo "  Warning: Missing results.json files, skipping t-test"
        continue
    fi
    
    # Run t-test
    python t_test.py "$progress_dir" "$no_progress_dir"
    echo ""
done

echo "========================================="
echo "T-Tests Complete"
echo "========================================="
echo ""

# ==================== Generate Excel Summary ====================

echo "========================================="
echo "Generating Excel Summary"
echo "========================================="
echo ""

python generate_excel_summary.py "${BASE_OUTPUT_DIR}"

echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo "All results saved to: ${BASE_OUTPUT_DIR}"
echo "Excel summary: ${BASE_OUTPUT_DIR}/t_test_summary.xlsx"
echo ""

