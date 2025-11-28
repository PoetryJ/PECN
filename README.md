# PECN
**Progress-Enhanced ControlNet** for text-conditioned human-object future frame prediction.

## Overview

Predict future frames in human-object interaction videos using text descriptions and current frames. Given frame 20 and an action description (e.g., "moving something from left to right"), the model predicts frame 21.

**Dataset**: Something-Something V2, filtered to three tasks:
- `move_object`: Horizontal/vertical movement
- `drop_object`: Falling/dropping dynamics  
- `cover_object`: Covering/placement actions

**Models**: Two baseline approaches:
- **ControlNet + Stable Diffusion**: Conditions SD on current frame
- **InstructPix2Pix**: Direct image-to-image translation

## Project Structure

```
PECN/
├── data/
│   ├── dataset.py                      # Filter & sample 3 tasks
│   ├── video_loader.py                 # Extract all frames (multi-threaded)
│   └── sthv2_dataset_hf.py             # HuggingFace Dataset adapter
├── train_controlnet.py                 # Official diffusers script (adapted)
├── train_instruct_pix2pix.py           # Official diffusers script (adapted)
├── train_controlnet.sh                 # ControlNet full training
├── train_instruct_pix2pix.sh           # InstructPix2Pix full training
├── test_train_controlnet.sh            # ControlNet quick test
└── test_train_instruct_pix2pix.sh      # InstructPix2Pix quick test
```

## Quick Start

### 1. Setup

```bash
# Set HuggingFace mirror (for faster downloads in China)
export HF_ENDPOINT=https://hf-mirror.com

# Install dependencies
pip install -r requirements.txt
```

**Note**: Add `export HF_ENDPOINT=https://hf-mirror.com` to `~/.bashrc` for permanent use.

### 2. Prepare Data

Manually download something-something-v2. I download it from https://aistudio.baidu.com/datasetdetail/143575. Unzip and organize data files like:

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
python -m data.video_loader
```


### 3. Test Training (Recommended First)

Quickly verify the training pipeline with minimal data (~2-5 minutes):

```bash
# ControlNet test
bash test_train_controlnet.sh

# InstructPix2Pix test
bash test_train_instruct_pix2pix.sh

# View logs
tensorboard --logdir=./test_output/
```

### 4. Full Training

**Train ControlNet:**
```bash
# ControlNet training
bash train_controlnet.sh
```

**Train InstructPix2Pix:**
```bash
# InstructPix2Pix training
bash train_instruct_pix2pix.sh
```


### 5. Testing and Sampling

Use the provided sampling scripts to generate predictions with your trained models.

**ControlNet Sampling:**
```bash
python sample_from_controlnet.py
```
- Loads the trained ControlNet model from `outputs/controlnet_128`.
- Randomly selects 3 validation samples.
- Generates predictions and saves them to `test_results_controlnet/`.

**InstructPix2Pix Sampling:**
```bash
python sample_from_pix2pix.py
```
- Loads the trained InstructPix2Pix UNet from `outputs/instruct_pix2pix_128/unet`.
- Generates predictions for the same 3 validation samples (using fixed random seed).
- Saves results to `test_results_pix2pix/`.

**Note:** Both scripts use a fixed random seed (42) to ensure you can compare results on the same validation samples.


## Training Configuration

### Experiment Settings

The experiment was conducted using a single NVIDIA RTX 5090 (32GB) gpu.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 128x128 | Good balance of quality and speed |
| Batch size | 8-16 | Per GPU |
| Gradient accumulation | 2 | Effective batch = 16-32 |
| Mixed precision | FP16 | 2x faster, 50% memory saving |
| Learning rate | 1e-5 (ControlNet)<br>5e-6 (InstructPix2Pix) | Lower for fine-tuning |
| Epochs | 20 | 20 minutes per model |

### Monitoring Commands

```bash
# Real-time progress
./monitor_training.sh

# TensorBoard (interactive)
tensorboard --logdir=outputs/ --port 6007

# GPU utilization
watch -n 1 nvidia-smi

# Training logs
tail -f outputs/controlnet_128_gpu0/train.log
tail -f outputs/instruct_pix2pix_128_gpu1/train.log
```


## Troubleshooting

| Issue | Solution |
|-------|----------|
| Network unreachable / Can't download models | 1. `export HF_ENDPOINT=https://hf-mirror.com` (see [HF-Mirror](https://hf-mirror.com/)) 2. In .py scripts, set 'os.environ["HF_ENDPOINT"] = HF_MIRROR' before 'from diffusers import xxxx' |
| Multi-GPU "CUDA illegal memory access" | Increase `--train_batch_size` to ≥4 per GPU, or use single GPU for smaller datasets |
| Out of memory | `--train_batch_size=1 --gradient_accumulation_steps=4 --mixed_precision="fp16"` |
| Slow training | `--mixed_precision="fp16"` (xformers optional, not required) |
| No samples | Run `python -m data.dataset` then `python -m data.video_loader` |

## Technical Details

**Data Pipeline:**
1. Filter ~220K videos → sample 3000 train + 300 val (3 tasks, 1000+100 per task)
2. Extract all frames from selected videos
3. Resize to 96×96 and 128×128, cache as PNG in separate folders
4. Training scripts load frame 20→21 pairs via HuggingFace Datasets

**Validation:**
- **ControlNet**: Uses `--validation_image` (local path) + `--validation_steps` (every N steps)
- **InstructPix2Pix**: Uses `--val_image_url` (local path or URL) + `--validation_epochs` (every N epochs)
- Results logged to TensorBoard for visual inspection

## Citation

Based on [diffusers examples](https://github.com/huggingface/diffusers/tree/main/examples) and [Something-Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something) dataset.
