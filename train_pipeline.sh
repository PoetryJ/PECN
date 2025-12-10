#!/bin/bash
# InstructPix2Pix Training Pipeline
# Trains 4 models with different configurations: frame21/frame25, with/without progress
# Resolution: 224x224, Epochs: 50, 1 GPU, No checkpointing

set -e

# Use HF-Mirror and specify GPUs
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# Set HuggingFace cache to project directory (save system disk space)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${SCRIPT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# ==================== Configuration ====================

# Common training parameters
DATA_DIR="./data/sthv2"
RESOLUTION=224
BATCH_SIZE=4           # Per GPU batch size (restored)
GRADIENT_ACCUM=4       # Effective batch size = 4 * 2 GPUs * 4 = 32
EPOCHS=40
LEARNING_RATE=5e-6     # Lower LR for InstructPix2Pix (fine-tuning)
INPUT_FRAME_IDX=20
NUM_GPUS=1

# Validation setup
VALIDATION_EPOCHS=10    # Validate every 10 epochs
NUM_VALIDATION_IMAGES=2  # Number of samples from val_filtered.json to use for validation

# No checkpointing (as requested)
# Set a very large value to disable intermediate checkpointing
# Final model will still be saved at the end of training
CHECKPOINTING_STEPS=999999999
CHECKPOINTS_TOTAL_LIMIT=0

# ==================== Parameter Combinations ====================

# Define all combinations
TASKS=("basic")
TARGET_FRAMES=(21 25)
USE_PROGRESS_OPTIONS=("true" "false")

# ==================== Helper Functions ====================

# Function to get output directory name
get_output_dir() {
    local task="$1"
    local target_frame="$2"
    local use_progress="$3"
    
    if [ "$target_frame" = "21" ]; then
        if [ "$task" = "basic" ]; then
            if [ "$use_progress" = "true" ]; then
                echo "outputs/instruct_pix2pix_224_progress"
            else
                echo "outputs/instruct_pix2pix_224"
            fi
        else
            if [ "$use_progress" = "true" ]; then
                echo "outputs/instruct_pix2pix_224_${task}_progress"
            else
                echo "outputs/instruct_pix2pix_224_${task}"
            fi
        fi
    else
        # frame25 uses _next5 suffix
        if [ "$task" = "basic" ]; then
            if [ "$use_progress" = "true" ]; then
                echo "outputs/instruct_pix2pix_224_progress_next5"
            else
                echo "outputs/instruct_pix2pix_224_next5"
            fi
        else
            if [ "$use_progress" = "true" ]; then
                echo "outputs/instruct_pix2pix_224_${task}_next5_progress"
            else
                echo "outputs/instruct_pix2pix_224_${task}_next5"
            fi
        fi
    fi
}

# ==================== Main Execution ====================

echo "=========================================="
echo "InstructPix2Pix Training Pipeline"
echo "=========================================="
echo "Configuration:"
echo "  GPU: 0 (1 GPU)"
echo "  Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRADIENT_ACCUM}"
echo "  Effective batch size: $((BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUM))"
echo "  Epochs: ${EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Validation: Every ${VALIDATION_EPOCHS} epoch(s)"
echo "  Checkpointing: Disabled"
echo ""
echo "Parameter combinations:"
echo "  Tasks: ${TASKS[@]}"
echo "  Target frames: ${TARGET_FRAMES[@]}"
echo "  Use progress: ${USE_PROGRESS_OPTIONS[@]}"
echo "Total combinations: $(( ${#TASKS[@]} * ${#TARGET_FRAMES[@]} * ${#USE_PROGRESS_OPTIONS[@]} ))"
echo "=========================================="
echo ""

# Check if frames directory exists
FRAMES_DIR="${DATA_DIR}/frames_${RESOLUTION}x${RESOLUTION}"
if [ ! -d "$FRAMES_DIR" ]; then
    echo "Error: Frames directory not found at $FRAMES_DIR"
    echo "Please extract frames first using: python -m data.video_loader"
    exit 1
fi

# Run training for each combination
combination_count=0
total_combinations=$((${#TASKS[@]} * ${#TARGET_FRAMES[@]} * ${#USE_PROGRESS_OPTIONS[@]}))

for task in "${TASKS[@]}"; do
    for target_frame in "${TARGET_FRAMES[@]}"; do
        for use_progress in "${USE_PROGRESS_OPTIONS[@]}"; do
            combination_count=$((combination_count + 1))
            
            # Get output directory
            output_dir=$(get_output_dir "$task" "$target_frame" "$use_progress")
            
            # Set progress flag
            if [ "$use_progress" = "true" ]; then
                PROGRESS_FLAG="--add_progress"
            else
                PROGRESS_FLAG=""
            fi
            
            echo ""
            echo "=========================================="
            echo "Training ${combination_count}/${total_combinations}"
            echo "=========================================="
            echo "Task: $task"
            echo "Target frame: $target_frame"
            echo "Use progress: $use_progress"
            echo "Output directory: $output_dir"
            echo "=========================================="
            echo ""
            
            # Create output directory
            mkdir -p "$output_dir"
            
            # Save training configuration
            cat > "$output_dir/train_config.txt" << EOF
Training started: $(date)
Model: InstructPix2Pix
Dataset: STHV2
GPU: 0 (1 GPU)
Resolution: ${RESOLUTION}x${RESOLUTION}
Batch size per GPU: ${BATCH_SIZE}
Gradient accumulation: ${GRADIENT_ACCUM}
Effective batch size: $((BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUM))
Epochs: ${EPOCHS}
Learning rate: ${LEARNING_RATE}
Mixed precision: FP16
Checkpointing: Disabled
Progress: ${use_progress}
Input frame index: ${INPUT_FRAME_IDX}
Target frame index: ${target_frame}
Task type: ${task}
EOF
            
            # Launch training
            echo "Starting training... (logs will be saved to ${output_dir}/train.log)"
            echo ""
            
            # Build accelerate launch command (single GPU mode)
            ACCELERATE_CMD=(
                accelerate launch
                --num_processes=1
                --mixed_precision=fp16
            )
            
            # Build training command
            TRAIN_CMD=(
                train_instruct_pix2pix.py
                --pretrained_model_name_or_path="timbrooks/instruct-pix2pix"
                --train_data_dir="$DATA_DIR"
                --resolution=$RESOLUTION
                --train_batch_size=$BATCH_SIZE
                --num_train_epochs=$EPOCHS
                --gradient_accumulation_steps=$GRADIENT_ACCUM
                --learning_rate=$LEARNING_RATE
                --lr_scheduler="constant"
                --lr_warmup_steps=500
                --output_dir="$output_dir"
                --checkpointing_steps=$CHECKPOINTING_STEPS
                --checkpoints_total_limit=$CHECKPOINTS_TOTAL_LIMIT
                --seed=42
                --report_to="tensorboard"
                --dataloader_num_workers=0
                --validation_epochs=$VALIDATION_EPOCHS
                --num_validation_images=$NUM_VALIDATION_IMAGES
                --input_frame_idx=$INPUT_FRAME_IDX
                --target_frame_idx=$target_frame
                --task=$task
            )
            
            # Add progress flag if enabled
            if [ -n "$PROGRESS_FLAG" ]; then
                TRAIN_CMD+=($PROGRESS_FLAG)
            fi
            
            # Run training (suppress set -e temporarily to continue on error)
            set +e
            "${ACCELERATE_CMD[@]}" "${TRAIN_CMD[@]}" 2>&1 | tee "$output_dir/train.log"
            train_exit_code=$?
            set -e
            
            if [ $train_exit_code -ne 0 ]; then
                echo ""
                echo "Warning: Training failed for combination (task=$task, frame=$target_frame, progress=$use_progress)"
                echo "Exit code: $train_exit_code"
                echo "Continuing with next combination..."
            else
                echo ""
                echo "Training completed successfully for: $output_dir"
                echo "Training finished: $(date)" >> "$output_dir/train_config.txt"
            fi
            
            echo ""
            echo "=========================================="
            echo ""
        done
    done
done

echo ""
echo "=========================================="
echo "Training Pipeline Complete!"
echo "=========================================="
echo "All models saved to: outputs/"
echo ""
echo "Trained models:"
for task in "${TASKS[@]}"; do
    for target_frame in "${TARGET_FRAMES[@]}"; do
        for use_progress in "${USE_PROGRESS_OPTIONS[@]}"; do
            output_dir=$(get_output_dir "$task" "$target_frame" "$use_progress")
            if [ -d "$output_dir" ]; then
                echo "  ✓ $output_dir"
            else
                echo "  ✗ $output_dir (failed)"
            fi
        done
    done
done
echo ""
echo "Pipeline finished: $(date)"
echo "=========================================="
