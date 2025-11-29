# PECN
**Progress-Enhanced InstructPix2Pix** for text-conditioned human-object future frame prediction.

## Overview

Predict future frames in human-object interaction videos using text descriptions and current frames. Given frame 20 (N) and an action description (e.g., "moving something from left to right"), the model predicts frame 21 (N + T).

**Dataset**: Something-Something V2, filtered to three tasks:
- `move_object`: Horizontal/vertical movement
- `drop_object`: Falling/dropping dynamics  
- `cover_object`: Covering/placement actions

**Model**: InstructPix2Pix for direct image-to-image translation with text instructions.

## Project Structure

```
PECN/
├── data/
│   ├── __init__.py
│   ├── dataset.py                      # Filter & sample 3 tasks
│   ├── video_loader.py                 # Extract all frames
│   ├── sthv2_dataset_hf.py             # HuggingFace Dataset adapter
│   └── sthv2/                          # STHV2 dataset
│       ├── annotations/                # Annotations
│       ├── videos/                     # Video files (not in git)
│       ├── frames_96x96/               # Extracted frames 96x96 (not in git)
│       └── frames_128x128/             # Extracted frames 128x128 (not in git)
├── test_result/                        # output of `sample_from_pix2pix.sh`
├── train_instruct_pix2pix.py           # InstructPix2Pix training script
├── train_instruct_pix2pix.sh           # InstructPix2Pix full training
├── test_train_instruct_pix2pix.sh      # InstructPix2Pix quick test
├── sample_from_pix2pix.py              # Sampling script for InstructPix2Pix
├── sample_from_pix2pix.sh              # Sampling script wrapper
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── .gitignore                          # Git ignore rules
```

**Note:** The following directories are excluded from git (see `.gitignore`):
- `data/sthv2/videos/` - Video files
- `data/sthv2/frames_*/` - Extracted frames
- `outputs/` - Training outputs and checkpoints
- `test_output/` - Test training outputs
- `.cache/` - HuggingFace model cache

## Quick Start

### 1. Setup

**System Requirements:**
- GPU: NVIDIA GPU with at least **~20GB VRAM** (recommended: 24GB+ for comfortable training)
- Python 3.8+
- CUDA-capable GPU

```bash
# Set HuggingFace mirror (for faster downloads in China)
export HF_ENDPOINT=https://hf-mirror.com

# Install dependencies
pip install -r requirements.txt
```

**Note**: Add `export HF_ENDPOINT=https://hf-mirror.com` to `~/.bashrc` for permanent use.

### 2. Prepare Data

**Note**: If you plan to use the progress-enhanced training feature (`--add_progress`), you must extract frames **after** filtering the dataset, as frame count information needs to be saved in the annotation files.

Manually download [Something-Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something). I download it from https://aistudio.baidu.com/datasetdetail/143575. Unzip and organize data files like:

```
PECN/
├── data/
│   ├── sthv2/
│   │   ├── annotations/
│   │   │   └── *.json
│   │   └── videos/
│   │       └── *.webm
│   └── ...
└── ...
```

```bash
# Filter to 3 tasks and sample (3000 train + 300 val samples)
python -m data.dataset

# Extract all frames in two resolutions (96x96 and 128x128)
# This will also save num_frames to annotations for progress calculation and update annotation files.
python -m data.video_loader
```


### 3. Test Training

Quickly verify the training pipeline with minimal data (~2-5 minutes):

```bash
# InstructPix2Pix test
bash test_train_instruct_pix2pix.sh

# View logs
tensorboard --logdir=./test_output/
```

### 4. Full Training

**Train InstructPix2Pix:**
```bash
# InstructPix2Pix training
bash train_instruct_pix2pix.sh
```


### 5. Testing and Sampling

Use the provided sampling scripts to generate predictions with your trained models.

**InstructPix2Pix Sampling:**
```bash
bash sample_from_pix2pix.sh
```

Or with custom parameters:
```bash
# Sample with custom inference parameters
MODEL_DIR=outputs/instruct_pix2pix_128 \
NUM_SAMPLES=10 \
NUM_INFERENCE_STEPS=50 \
IMAGE_GUIDANCE_SCALE=1.2 \
OUTPUT_DIR=results_custom \
bash sample_from_pix2pix.sh
```

**Available environment variables for sampling:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `outputs/instruct_pix2pix_128` | Path to trained model |
| `FRAMES_DIR` | `data/sthv2/frames_128x128` | Directory with extracted frames |
| `VAL_ANNOTATIONS` | `data/sthv2/annotations/val_filtered.json` | Validation annotations |
| `OUTPUT_DIR` | `test_result/baseline` | Where to save results |
| `INPUT_FRAME_IDX` | `20` | Input frame index |
| `TARGET_FRAME_IDX` | `21` | Target/GT frame index |
| `NUM_SAMPLES` | `3` | Number of samples to generate |
| `SEED` | `42` | Random seed for reproducibility |
| `NUM_INFERENCE_STEPS` | `20` | Diffusion steps |
| `IMAGE_GUIDANCE_SCALE` | `1.5` | Image guidance strength |

**Note:** The sampling script uses a fixed random seed (42) by default to ensure reproducible results.


## Advanced Training Options

**Important**: To use progress-enhanced training, you must have run `python -m data.video_loader` to extract frames, which will automatically add `num_frames` information to the annotation files.

**1. Progress-Enhanced Training**

Add action completion percentage to prompts for better temporal understanding.

Open the `train_instruct_pix2pix.sh` script and set:

``` shell
PROGRESS=True
```

This modifies the instruction to include progress information:
- **Without progress**: `"Generate a future frame of this action: moving something from left to right"`
- **With progress**: `"Generate a future frame of this action: moving something from left to right. And 60% of the action has been completed."`

The progress is calculated as: `progress = int((input_frame_idx / total_frames) * 10) * 10`

**2. Custom Frame Prediction**

To specify which frames to use as input/target, open the `train_instruct_pix2pix.sh` script and set:

```shell
# Predict frame 25 from frame 20 (further future prediction)
INPUT_FRAME_IDX=20
TARGET_FRAME_IDX=25
```

## Training Configuration

### Experiment Settings

The experiment was conducted using a single NVIDIA RTX 5090 (32GB) gpu.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 128x128 | Good balance of quality and speed |
| Batch size | 8-16 | Per GPU |
| Gradient accumulation | 2 | Effective batch = 16-32 |
| Mixed precision | FP16 | 2x faster, 50% memory saving |
| Learning rate | 5e-6 | Lower for fine-tuning |
| Epochs | 20 | 20 minutes per model |

### Monitoring Commands

```bash
# TensorBoard (interactive)
tensorboard --logdir=outputs/ --port 6007

# GPU utilization
watch -n 1 nvidia-smi

# Training logs
tail -f outputs/instruct_pix2pix_128/train.log
```

And you can find validation images generated during training in {output_dir}/validation/.


## Technical Details

**Data Pipeline:**
1. Filter ~220K videos → sample 3000 train + 300 val (3 tasks, 1000+100 per task)
2. Extract all frames from selected videos
3. Resize to 96×96 and 128×128, cache as PNG in separate folders
4. Training scripts load frame pairs via HuggingFace Datasets (configurable frame indices)

**Validation:**
- Automatically selects samples from `val_filtered.json` and generates predictions
- Validation runs every N epochs (configurable via `--validation_epochs`)
- Results saved to `{output_dir}/validation/` as comparison images
- Results also logged to TensorBoard for visual inspection



## Troubleshooting

| Issue | Solution |
|-------|----------|
| Network unreachable / Can't download models | 1. `export HF_ENDPOINT=https://hf-mirror.com` (see [HF-Mirror](https://hf-mirror.com/)) 2. In .py scripts, set 'os.environ["HF_ENDPOINT"] = HF_MIRROR' before 'from diffusers import xxxx' |
| Multi-GPU "CUDA illegal memory access" | Increase `--train_batch_size` to ≥4 per GPU, or use single GPU for smaller datasets |
| Out of memory | `--train_batch_size=1 --gradient_accumulation_steps=4 --mixed_precision="fp16"` |
| Slow training | `--mixed_precision="fp16"` (xformers optional, not required) |
| No samples | Run `python -m data.dataset` then `python -m data.video_loader` |


## Citation

Based on [diffusers examples](https://github.com/huggingface/diffusers/tree/main/examples) and [Something-Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something) dataset.
