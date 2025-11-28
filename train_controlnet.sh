#!/bin/bash
# ControlNet full training on GPU 0
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
OUTPUT_DIR="./outputs/controlnet_128"
DATA_DIR="./data/sthv2"
RESOLUTION=128
BATCH_SIZE=8           # Adjust based on GPU memory (8-16 for RTX 5090)
GRADIENT_ACCUM=2       # Effective batch size = 8 * 2 = 16
EPOCHS=20
LEARNING_RATE=1e-5
CHECKPOINTING_STEPS=500
VALIDATION_PROMPT="Pushing something from right to left"
VALIDATION_IMAGE="${DATA_DIR}/frames_128x128/100000_frame_00020.png"

echo "=========================================="
echo "ControlNet Full Training (GPU 0)"
echo "=========================================="
echo "Configuration:"
echo "  GPU: 0 (NVIDIA GeForce RTX 5090)"
echo "  Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRADIENT_ACCUM}"
echo "  Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUM))"
echo "  Epochs: ${EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save training configuration
cat > "$OUTPUT_DIR/train_config.txt" << EOF
Training started: $(date)
Model: ControlNet + Stable Diffusion 1.5
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
    train_controlnet.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
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
    --validation_image="$VALIDATION_IMAGE" \
    --validation_prompt="$VALIDATION_PROMPT" \
    --validation_steps=500 \
    --seed=42 \
    --report_to="tensorboard" \
    --dataloader_num_workers=4 \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "=========================================="
echo "ControlNet Training Completed!"
echo "=========================================="
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "To view training logs:"
echo "  tensorboard --logdir=$OUTPUT_DIR/logs --port 6007"
echo ""
echo "Training finished: $(date)" >> "$OUTPUT_DIR/train_config.txt"

