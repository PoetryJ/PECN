#!/bin/bash
# InstructPix2Pix full training on GPU 0
# Resolution: 128x128, Samples: 3000, Epochs: 10
# Estimated time: 5-8 hours on RTX 5090

set -e

# Use HF-Mirror and specify GPU 0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1 # bug: online mode will cause the training to fail
export CUDA_VISIBLE_DEVICES=0

# Set HuggingFace cache to project directory (save system disk space)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${SCRIPT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# Configuration
OUTPUT_DIR="./outputs/instruct_pix2pix_128"
DATA_DIR="./data/sthv2"
RESOLUTION=128
BATCH_SIZE=8           # Adjust based on GPU memory (8-16 for RTX 5090)
GRADIENT_ACCUM=2       # Effective batch size = 8 * 2 = 16
EPOCHS=20
LEARNING_RATE=5e-6     # Lower LR for InstructPix2Pix (fine-tuning)
CHECKPOINTING_STEPS=500

# Validation setup
VALIDATION_IMAGE="${DATA_DIR}/frames_128x128/100000_frame_00020.png"
VALIDATION_PROMPT="pushing color pencils from right to left"
VALIDATION_EPOCHS=5    # Validate every 5 epoch
NUM_VALIDATION_IMAGES=4

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
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Check if validation image exists
if [ ! -f "$VALIDATION_IMAGE" ]; then
    echo "Warning: Validation image not found at $VALIDATION_IMAGE"
    echo "Training will proceed without validation."
    VALIDATION_IMAGE=""
fi

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
    ${VALIDATION_IMAGE:+--val_image_url="$VALIDATION_IMAGE"} \
    ${VALIDATION_IMAGE:+--validation_prompt="$VALIDATION_PROMPT"} \
    ${VALIDATION_IMAGE:+--validation_epochs=$VALIDATION_EPOCHS} \
    ${VALIDATION_IMAGE:+--num_validation_images=$NUM_VALIDATION_IMAGES} \
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

