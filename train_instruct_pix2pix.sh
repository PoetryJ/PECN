#!/bin/bash
# InstructPix2Pix full training on GPU 0
# Resolution: 128x128, Samples: 3000, Epochs: 10
# Estimated time: 5-8 hours on RTX 5090

set -e

# Use HF-Mirror and specify GPU 0
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# Set HuggingFace cache to project directory (save system disk space)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${SCRIPT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# Configuration
OUTPUT_DIR="./outputs/instruct_pix2pix_128_progress_next5"
DATA_DIR="./data/sthv2"
RESOLUTION=128
BATCH_SIZE=8           # Adjust based on GPU memory (8-16 for RTX 5090)
GRADIENT_ACCUM=2       # Effective batch size = 8 * 2 = 16
EPOCHS=20
LEARNING_RATE=5e-6     # Lower LR for InstructPix2Pix (fine-tuning)
CHECKPOINTING_STEPS=500
PROGRESS=True
INPUT_FRAME_IDX=20
TARGET_FRAME_IDX=25
TASK="basic"  # Options: basic, backforth

# Validation setup
VALIDATION_EPOCHS=5    # Validate every 5 epoch
NUM_VALIDATION_IMAGES=4  # Number of samples from val_filtered.json to use for validation

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_frame_idx=*)
            INPUT_FRAME_IDX="${1#*=}"
            shift
            ;;
        --target_frame_idx=*)
            TARGET_FRAME_IDX="${1#*=}"
            shift
            ;;
        --add_progress)
            PROGRESS="True"
            shift
            ;;
        --no_progress)
            PROGRESS="False"
            shift
            ;;
        --task=*)
            TASK="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate task value
if [ "${TASK}" != "basic" ] && [ "${TASK}" != "backforth" ]; then
    echo "Error: --task must be 'basic' or 'backforth', got '${TASK}'"
    exit 1
fi

# Set progress flag based on PROGRESS value
if [ "${PROGRESS}" = "True" ] || [ "${PROGRESS}" = "true" ]; then
    PROGRESS_FLAG="--add_progress"
else
    PROGRESS_FLAG=""
fi

echo "=========================================="
echo "InstructPix2Pix Full Training (GPU 0)"
echo "=========================================="
echo "Configuration:"
echo "  GPU: 0 (NVIDIA GeForce RTX 5090)"
echo "  Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRADIENT_ACCUM}"
echo "  Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUM))"
echo "  Epochs: ${EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Validation: Every ${VALIDATION_EPOCHS} epoch(s)"
if [ -n "$PROGRESS_FLAG" ]; then
    echo "  Progress enhancement: Enabled"
else
    echo "  Progress enhancement: Disabled"
fi
echo "  Input frame index: ${INPUT_FRAME_IDX}"
echo "  Target frame index: ${TARGET_FRAME_IDX}"
echo "  Task type: ${TASK}"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save training configuration
cat > "$OUTPUT_DIR/train_config.txt" << EOF
Training started: $(date)
Model: InstructPix2Pix
Dataset: STHV2 (3000 samples, 3 tasks)
GPU: 0 (CUDA_VISIBLE_DEVICES=0)
Resolution: ${RESOLUTION}x${RESOLUTION}
Batch size: ${BATCH_SIZE}
Gradient accumulation: ${GRADIENT_ACCUM}
Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUM))
Epochs: ${EPOCHS}
Learning rate: ${LEARNING_RATE}
Mixed precision: FP16
Checkpointing every: ${CHECKPOINTING_STEPS} steps
Progress: ${PROGRESS}
Input frame index: ${INPUT_FRAME_IDX}
Target frame index: ${TARGET_FRAME_IDX}
Task type: ${TASK}
EOF

# Launch training
echo "Starting training... (logs will be saved to ${OUTPUT_DIR}/train.log)"
echo ""

accelerate launch \
    --num_processes=1 \
    --mixed_precision=fp16 \
    train_instruct_pix2pix.py \
    --pretrained_model_name_or_path="timbrooks/instruct-pix2pix" \
    --train_data_dir="$DATA_DIR" \
    --resolution=$RESOLUTION \
    --train_batch_size=$BATCH_SIZE \
    --num_train_epochs=$EPOCHS \
    --gradient_accumulation_steps=$GRADIENT_ACCUM \
    --learning_rate=$LEARNING_RATE \
    --lr_scheduler="constant" \
    --lr_warmup_steps=500 \
    --output_dir="$OUTPUT_DIR" \
    --checkpointing_steps=$CHECKPOINTING_STEPS \
    --checkpoints_total_limit=3 \
    --seed=42 \
    --report_to="tensorboard" \
    --dataloader_num_workers=0 \
    --validation_epochs=$VALIDATION_EPOCHS \
    --num_validation_images=$NUM_VALIDATION_IMAGES \
    $PROGRESS_FLAG \
    ${INPUT_FRAME_IDX:+--input_frame_idx=$INPUT_FRAME_IDX} \
    ${TARGET_FRAME_IDX:+--target_frame_idx=$TARGET_FRAME_IDX} \
    --task=$TASK \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "=========================================="
echo "InstructPix2Pix Training Completed!"
echo "=========================================="
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "To view training logs:"
echo "  tensorboard --logdir=$OUTPUT_DIR/logs --port 6007"
echo ""
echo "Training finished: $(date)" >> "$OUTPUT_DIR/train_config.txt"

