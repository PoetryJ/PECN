#!/bin/bash
# Test script for InstructPix2Pix training with minimal data (SINGLE GPU)
# More stable for quick testing

set -e  # Exit on error

# Use HF-Mirror for faster downloads in China
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0

echo "=========================================="
echo "Testing InstructPix2Pix Training Pipeline (Single GPU)"
echo "=========================================="

# Configuration
OUTPUT_DIR="./test_output/instruct_pix2pix_single"
DATA_DIR="./data/sthv2"
RESOLUTION=96  # Use smaller resolution for faster testing
MAX_SAMPLES=10  # Only use 10 samples for testing
MAX_STEPS=5     # Only train for 5 steps
BATCH_SIZE=1    # Small batch size
GRADIENT_ACCUM=2

# Pretrained model (using InstructPix2Pix base model)
PRETRAINED_MODEL="timbrooks/instruct-pix2pix"

# Validation setup (now supports local paths)
VALIDATION_IMAGE="${DATA_DIR}/frames_96x96/150479_frame_00020.png"
VALIDATION_PROMPT="pushing color pencils from right to left"
VALIDATION_EPOCHS=1  # Validate every epoch

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
    echo "Warning: Validation image not found at $VALIDATION_IMAGE"
    echo "Validation will be skipped. Run 'python -m data.video_loader --test' to extract frames."
    VALIDATION_IMAGE=""
fi

# Launch training with accelerate (single GPU)
accelerate launch \
    --num_processes=1 \
    --mixed_precision=fp16 \
    train_instruct_pix2pix.py \
    --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
    --train_data_dir="$DATA_DIR" \
    --resolution=$RESOLUTION \
    --train_batch_size=$BATCH_SIZE \
    --max_train_samples=$MAX_SAMPLES \
    --max_train_steps=$MAX_STEPS \
    --gradient_accumulation_steps=$GRADIENT_ACCUM \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir="$OUTPUT_DIR" \
    --checkpointing_steps=5 \
    --seed=42 \
    --report_to="tensorboard" \
    --dataloader_num_workers=2 \
    ${VALIDATION_IMAGE:+--val_image_url="$VALIDATION_IMAGE"} \
    ${VALIDATION_IMAGE:+--validation_prompt="$VALIDATION_PROMPT"} \
    ${VALIDATION_IMAGE:+--validation_epochs=$VALIDATION_EPOCHS} \
    ${VALIDATION_IMAGE:+--num_validation_images=1}

echo ""
echo "=========================================="
echo "InstructPix2Pix Test Training Completed!"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "To view training logs:"
echo "  tensorboard --logdir=$OUTPUT_DIR/logs"
echo ""

