#!/bin/bash
# Test script for ControlNet training with minimal data (SINGLE GPU)
# More stable for quick testing

set -e  # Exit on error

# Use HF-Mirror for faster downloads in China
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0

echo "=========================================="
echo "Testing ControlNet Training Pipeline (Single GPU)"
echo "=========================================="

# Configuration
OUTPUT_DIR="./test_output/controlnet_single"
DATA_DIR="./data/sthv2"
RESOLUTION=96  # Use smaller resolution for faster testing
MAX_SAMPLES=10  # Only use 10 samples for testing
MAX_STEPS=5     # Only train for 5 steps
BATCH_SIZE=1    # Small batch size
GRADIENT_ACCUM=2

# Pretrained model (using SD 1.5 base model)
PRETRAINED_MODEL="runwayml/stable-diffusion-v1-5"

# Validation setup (use first training sample for quick validation)
VALIDATION_IMAGE="${DATA_DIR}/frames_96x96/150479_frame_00020.png"
VALIDATION_PROMPT="pushing color pencils from right to left"

# Clean previous test output
if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning previous test output..."
    rm -rf "$OUTPUT_DIR"
fi

echo ""
echo "Training Configuration:"
echo "  - GPUs: 1 (GPU 0)"
echo "  - Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "  - Max samples: ${MAX_SAMPLES}"
echo "  - Max steps: ${MAX_STEPS}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Gradient accumulation: ${GRADIENT_ACCUM}"
echo "  - Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUM))"
echo "  - Output directory: ${OUTPUT_DIR}"
echo ""

# Check if validation image exists
if [ ! -f "$VALIDATION_IMAGE" ]; then
    echo "Error: Validation image not found at $VALIDATION_IMAGE"
    echo "Please run 'python -m data.video_loader --test' first to extract frames."
    exit 1
fi

# Launch training with accelerate (single GPU)
accelerate launch \
    --num_processes=1 \
    --mixed_precision=fp16 \
    train_controlnet.py \
    --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
    --train_data_dir="$DATA_DIR" \
    --resolution=$RESOLUTION \
    --train_batch_size=$BATCH_SIZE \
    --max_train_samples=$MAX_SAMPLES \
    --max_train_steps=$MAX_STEPS \
    --gradient_accumulation_steps=$GRADIENT_ACCUM \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir="$OUTPUT_DIR" \
    --validation_image="$VALIDATION_IMAGE" \
    --validation_prompt="$VALIDATION_PROMPT" \
    --validation_steps=3 \
    --num_validation_images=1 \
    --checkpointing_steps=5 \
    --seed=42 \
    --report_to="tensorboard" \
    --dataloader_num_workers=2

echo ""
echo "=========================================="
echo "ControlNet Test Training Completed!"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "To view training logs:"
echo "  tensorboard --logdir=$OUTPUT_DIR/logs"
echo ""

