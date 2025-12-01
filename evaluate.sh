#!/bin/bash

# InstructPix2Pix Sampling Script
# This script samples from a trained InstructPix2Pix model

set -e

# ==================== Configuration ====================

# Model and data paths
MODEL_DIR="${MODEL_DIR:-outputs/instruct_pix2pix_128}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-timbrooks/instruct-pix2pix}"
FRAMES_DIR="${FRAMES_DIR:-data/sthv2/frames_128x128}"
VAL_ANNOTATIONS="${VAL_ANNOTATIONS:-data/sthv2/annotations/val_filtered.json}"
OUTPUT_DIR="${OUTPUT_DIR:-test_result/instruct_pix2pix_128}"

# Frame indices
INPUT_FRAME_IDX="${INPUT_FRAME_IDX:-20}"
TARGET_FRAME_IDX="${TARGET_FRAME_IDX:-21}"

# Sampling parameters
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-25}"
IMAGE_GUIDANCE_SCALE="${IMAGE_GUIDANCE_SCALE:-1.5}"

# ==================== Validation ====================

echo "========================================="
echo "InstructPix2Pix Sampling Configuration"
echo "========================================="
echo "Model directory: $MODEL_DIR"
echo "Pretrained model: $PRETRAINED_MODEL"
echo "Frames directory: $FRAMES_DIR"
echo "Val annotations: $VAL_ANNOTATIONS"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Input frame index: $INPUT_FRAME_IDX"
echo "Target frame index: $TARGET_FRAME_IDX"
echo ""
echo "Number of samples: $NUM_SAMPLES"
echo "Random seed: $SEED"
echo "Inference steps: $NUM_INFERENCE_STEPS"
echo "Image guidance scale: $IMAGE_GUIDANCE_SCALE"
echo "========================================="
echo ""

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Warning: Model directory not found at $MODEL_DIR"
    echo "Will attempt to download from pretrained model: $PRETRAINED_MODEL"
fi

# Check if frames directory exists
if [ ! -d "$FRAMES_DIR" ]; then
    echo "Error: Frames directory not found at $FRAMES_DIR"
    echo "Please run: python -m data.video_loader first"
    exit 1
fi

# Check if validation annotations exist
if [ ! -f "$VAL_ANNOTATIONS" ]; then
    echo "Error: Validation annotations not found at $VAL_ANNOTATIONS"
    echo "Please run: python -m data.dataset first"
    exit 1
fi

# ==================== Run Sampling ====================

echo "Starting InstructPix2Pix sampling..."
echo ""

python sample_from_pix2pix.py \
    --model_dir="$MODEL_DIR" \
    --pretrained_model="$PRETRAINED_MODEL" \
    --frames_dir="$FRAMES_DIR" \
    --val_annotations="$VAL_ANNOTATIONS" \
    --output_dir="$OUTPUT_DIR" \
    --input_frame_idx=$INPUT_FRAME_IDX \
    --target_frame_idx=$TARGET_FRAME_IDX \
    --num_samples=$NUM_SAMPLES \
    --seed=$SEED \
    --num_inference_steps=$NUM_INFERENCE_STEPS \
    --image_guidance_scale=$IMAGE_GUIDANCE_SCALE

echo ""
echo "========================================="
echo "Sampling complete! Results saved to $OUTPUT_DIR"
echo "========================================="
echo ""

# ==================== Calculate SSIM and PSNR ====================

echo "========================================="
echo "Calculating SSIM and PSNR metrics"
echo "========================================="
echo "Data directory: $OUTPUT_DIR"
echo ""

python ssim.py --data_dir="$OUTPUT_DIR"

echo ""
echo "========================================="
echo "Evaluation complete! Metrics saved to $OUTPUT_DIR/results.json"
echo "========================================="

