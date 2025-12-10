# Progress-Aware Video Frame Prediction with Cross-Modal Temporal Guidance

A framework for text-conditioned human-object future frame prediction using progress-aware cross-modal guidance.

## Overview

Predict future frames in human-object interaction videos using text descriptions and current frames. Given frame 20 and an action description (e.g., "moving something from left to right"), the model predicts frame 21.

**Dataset**: Something-Something V2, filtered to four task categories:
- `move_object`: Horizontal/vertical movement
- `drop_object`: Falling/dropping dynamics  
- `cover_object`: Covering/placement actions
- `back_and_forth`: Cyclic actions (rolling, plugging/unplugging, throwing/catching)

**Model**: Fine-tuned InstructPix2Pix for direct image-to-image translation with text instructions, optionally enhanced with progress-aware prompts.

## Project Structure

```
PECN/
├── data/
│   ├── __init__.py
│   ├── dataset.py                      # Filter & sample 4 tasks
│   ├── video_loader.py                 # Extract all frames
│   ├── sthv2_dataset_hf.py             # HuggingFace Dataset adapter
│   └── sthv2/                          # STHV2 dataset
│       ├── annotations/                # Annotations (train_filtered.json, val_filtered.json, etc.)
│       ├── videos/                     # Video files (not in git)
│       ├── frames_96x96/               # Extracted frames 96x96 (not in git)
│       ├── frames_128x128/             # Extracted frames 128x128 (not in git)
│       └── frames_224x224/             # Extracted frames 224x224 (not in git)
├── progress_evaluator/                 # Progress estimation module (PAC-Net)
├── outputs/                            # Trained models (not in git)
├── test_result/                        # Evaluation results
├── train_instruct_pix2pix.py           # InstructPix2Pix training script
├── train_instruct_pix2pix.sh           # Single model training
├── train_pipeline.sh                   # Batch training pipeline (4 models)
├── evaluate.sh                         # Single model evaluation
├── evaluate_pipeline.sh                # Batch evaluation pipeline
├── sample_from_pix2pix.py              # Sampling script
├── ssim.py                             # SSIM and PSNR calculation
├── t_test.py                           # Statistical t-test analysis
├── generate_excel_summary.py           # Generate Excel summary from results
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── .gitignore                          # Git ignore rules
```

**Note:** The following directories are excluded from git (see `.gitignore`):
- `data/sthv2/videos/` - Video files
- `data/sthv2/frames_*/` - Extracted frames
- `outputs/` - Training outputs and checkpoints
- `test_result/` - Evaluation results
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
# Filter to 4 tasks and sample (2000 per task for training, 100 per task for validation)
# Note: back_and_forth has only 1962 samples, so total training set is 7962 samples
python -m data.dataset

# Extract all frames in multiple resolutions (96x96, 128x128, 224x224)
# This will also save num_frames to annotations for progress calculation
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

**Train a single InstructPix2Pix model:**
```bash
# Single model training (customize parameters in the script)
bash train_instruct_pix2pix.sh
```

**Train multiple models with pipeline:**
```bash
# Train 4 models: frame21/frame25 × with/without progress
# All models are trained on the same mixed dataset (basic + backforth)
bash train_pipeline.sh
```

The pipeline will train:
- `instruct_pix2pix_224` - frame 21, without progress
- `instruct_pix2pix_224_progress` - frame 21, with progress
- `instruct_pix2pix_224_next5` - frame 25, without progress
- `instruct_pix2pix_224_progress_next5` - frame 25, with progress

**Train Progress-Aware Cross-modal Network (PAC-Net):**

PAC-Net estimates temporal progress from video frames and action descriptions. It is trained using dynamic temporal cropping for self-supervised progress targets.

```bash
# Regression-based training (recommended)
cd progress_evaluator
python train_reg.py \
    --train_data_json ../data/sthv2/annotations/train_filtered.json \
    --val_data_json ../data/sthv2/annotations/val_filtered.json \
    --num_frames 20 \
    --hidden_dim 512 \
    --batch_size 8 \
    --epochs 30 \
    --lr 5e-5 \
    --weight_decay 5e-5 \
    --num_workers 8 \
    --save_dir ./checkpoints
```

**Training details:**
- **Dynamic sampling**: For each video, randomly sample a progress ratio (0.1-0.9), then uniformly sample 20 frames from the beginning to that progress point
- **Self-supervised targets**: Progress ground truth is computed as the physical time ratio (end_frame / total_frames), leveraging SSv2's natural temporal alignment
- **Architecture**: ResNet50 frame encoder + Transformer temporal encoder + CLIP text encoder + cross-modal fusion
- **Loss**: Smooth L1 regression loss, evaluated with MAE (Mean Absolute Error)
- **Optimization**: AdamW with layered learning rates (backbone: 0.2×lr, head: 1×lr), cosine annealing scheduler

The trained model will be saved to `progress_evaluator/checkpoints/best_student.pth` and can be used to estimate progress for InstructPix2Pix training with `--add_progress` flag.

**Alternative: Classification-based training:**
```bash
# Classification-based training (for comparison)
python train_class.py \
    --train_data_json ../data/sthv2/annotations/train_filtered.json \
    --val_data_json ../data/sthv2/annotations/val_filtered.json \
    --num_frames 20 \
    --hidden_dim 512 \
    --num_classes 101 \
    --batch_size 8 \
    --epochs 30 \
    --lr 5e-5 \
    --save_dir ./checkpoints
```

### 5. Testing and Evaluation

**Single model evaluation:**
```bash
# Evaluate a single model
bash evaluate.sh
```

The script will:
1. Generate predictions using the trained model
2. Automatically calculate SSIM and PSNR metrics
3. Save results to `{OUTPUT_DIR}/results.json`

**Batch evaluation pipeline:**
```bash
# Evaluate all models on both basic and backforth validation sets
# This performs 8 evaluations: 4 models × 2 validation sets
bash evaluate_pipeline.sh 42
```

The pipeline will:
1. Evaluate each trained model on both `val_filtered.json` (basic) and `val_filtered_backforth.json` (backforth)
2. Perform t-tests comparing models with/without progress
3. Generate Excel summary with all results

**Custom evaluation parameters:**
```bash
# Single model with custom parameters
MODEL_DIR=outputs/instruct_pix2pix_224 \
FRAMES_DIR=data/sthv2/frames_224x224 \
NUM_SAMPLES=100 \
NUM_INFERENCE_STEPS=25 \
IMAGE_GUIDANCE_SCALE=1.5 \
OUTPUT_DIR=test_result/instruct_pix2pix_224 \
bash evaluate.sh
```

**Available environment variables for `evaluate.sh`:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `outputs/instruct_pix2pix_224` | Path to trained model |
| `FRAMES_DIR` | `data/sthv2/frames_224x224` | Directory with extracted frames |
| `VAL_ANNOTATIONS` | `data/sthv2/annotations/val_filtered.json` | Validation annotations |
| `OUTPUT_DIR` | `test_result/instruct_pix2pix_224` | Where to save results |
| `INPUT_FRAME_IDX` | `20` | Input frame index |
| `TARGET_FRAME_IDX` | `21` | Target/GT frame index |
| `NUM_SAMPLES` | `100` | Number of samples to generate |
| `SEED` | `42` | Random seed for reproducibility |
| `NUM_INFERENCE_STEPS` | `25` | Diffusion steps |
| `IMAGE_GUIDANCE_SCALE` | `1.5` | Image guidance strength |
| `USE_PROGRESS_ESTIMATOR` | `false` | Use progress estimator for prompts |

**Output Structure:**
```
{OUTPUT_DIR}/
├── target/              # Ground truth images
├── output/              # Generated images
├── comparison/          # Comparison grids (Input | GT | Output)
├── results.json         # SSIM and PSNR metrics per sample
└── t_test_results.json # T-test results (if using pipeline)
```

**Batch evaluation output:**
```
test_result/seed_{SEED}/
├── instruct_pix2pix_224_eval_basic/
├── instruct_pix2pix_224_eval_backforth/
├── instruct_pix2pix_224_progress_eval_basic/
├── instruct_pix2pix_224_progress_eval_backforth/
├── ... (8 total evaluation directories)
└── t_test_summary.xlsx  # Excel summary of all results
```

**Note:** The evaluation script uses a fixed random seed (42) by default to ensure reproducible results.


## Advanced Training Options

**Important**: To use progress-enhanced training, you must have:
1. Run `python -m data.video_loader` to extract frames and add `num_frames` information to annotation files
2. Trained PAC-Net model (or use the provided checkpoint in `progress_evaluator/checkpoints/`)

**1. Progress-Enhanced Training**

Add action completion percentage to prompts for better temporal understanding, especially for cyclic back-and-forth actions.

In `train_instruct_pix2pix.sh` or `train_pipeline.sh`, set:

```shell
PROGRESS=True  # or use --add_progress flag
```

This modifies the instruction to include progress information:
- **Without progress**: `"Generate a future frame of this action: moving something from left to right"`
- **With progress**: `"Generate a future frame of this action: moving something from left to right. And 60% of the action has been completed."`

The progress is calculated as: `progress = int((input_frame_idx / total_frames) * 100)`

**2. Custom Frame Prediction**

To specify which frames to use as input/target, modify the scripts:

```shell
# Predict frame 21 from frame 20 (immediate next frame)
INPUT_FRAME_IDX=20
TARGET_FRAME_IDX=21

# Predict frame 25 from frame 20 (further future prediction)
INPUT_FRAME_IDX=20
TARGET_FRAME_IDX=25
```

## Training Configuration

### Experiment Settings

The experiments were conducted using a single GPU (single GPU mode to avoid DDP issues).

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 224x224 | Final resolution used in experiments |
| Batch size | 4 | Per GPU |
| Gradient accumulation | 4 | Effective batch size = 16 |
| Mixed precision | FP16 | Memory saving and faster training |
| Learning rate | 5e-6 | Lower for fine-tuning |
| Epochs | 40 | Per model |
| Training samples | 7962 | 4 tasks × ~2000 samples each (backforth has 1962) |
| Validation | Every 10 epochs | 2 samples from val_filtered.json |

### PAC-Net Training Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | ResNet50 + Transformer + CLIP | Cross-modal fusion |
| Input frames | 20 | Uniformly sampled from video start to random progress point |
| Hidden dimension | 512 | Shared feature space |
| Batch size | 8 | Per GPU |
| Learning rate | 5e-5 | Layered LR (backbone: 0.2×, head: 1×) |
| Weight decay | 5e-5 | Regularization |
| Epochs | 30 | With cosine annealing scheduler |
| Loss function | Smooth L1 | Regression-based (recommended) |
| Evaluation metric | MAE | Mean Absolute Error in percentage |
| Training strategy | Dynamic temporal cropping | Self-supervised progress targets |

### Monitoring Commands

```bash
# TensorBoard (interactive)
tensorboard --logdir=outputs/ --port 6007

# GPU utilization
watch -n 1 nvidia-smi

# Training logs (adjust path based on model)
tail -f outputs/instruct_pix2pix_224/train.log
```

Validation images generated during training are saved in `{output_dir}/validation/` as comparison grids showing original and generated frames.


## Technical Details

**Data Pipeline:**
1. Filter ~220K videos → sample 2000 train + 100 val per task (4 tasks: move, drop, cover, back_and_forth)
2. Total training set: 7962 samples (back_and_forth has 1962 instead of 2000)
3. Extract all frames from selected videos
4. Resize to 96×96, 128×128, and 224×224, cache as PNG in separate folders
5. Training scripts load frame pairs via HuggingFace Datasets (configurable frame indices)

**Training:**
- **InstructPix2Pix**: All models are trained on the same mixed dataset (`train_filtered.json` containing all 4 tasks)
- Models are trained with/without progress hints and for different target frames (21 vs 25)
- Validation runs every 10 epochs, using samples from `val_filtered.json`
- Results saved to `{output_dir}/validation/` as comparison images
- Results also logged to TensorBoard for visual inspection

- **PAC-Net**: Trained on the same filtered dataset using dynamic temporal cropping
- For each training sample, randomly selects a progress ratio (0.1-0.9) and samples 20 frames from video start to that point
- Progress ground truth is computed as physical time ratio (end_frame / total_frames)
- Uses cross-modal fusion of video (ResNet50 + Transformer) and text (CLIP) features
- Trained with regression loss (Smooth L1) for continuous progress estimation

**Evaluation:**
- Each trained model is evaluated on both validation sets:
  - `val_filtered.json` (mixed basic tasks, 100 samples)
  - `val_filtered_backforth.json` (back-and-forth tasks, 100 samples)
- Metrics: SSIM and PSNR calculated per sample
- Statistical analysis: t-tests comparing with/without progress models
- Results aggregated in Excel summary for easy comparison



## Troubleshooting

| Issue | Solution |
|-------|----------|
| Network unreachable / Can't download models | 1. `export HF_ENDPOINT=https://hf-mirror.com` (see [HF-Mirror](https://hf-mirror.com/)) 2. In .py scripts, set 'os.environ["HF_ENDPOINT"] = HF_MIRROR' before 'from diffusers import xxxx' |
| Multi-GPU "CUDA illegal memory access" | Use single GPU mode (set `NUM_GPUS=1` and `CUDA_VISIBLE_DEVICES=0`) |
| Out of memory | Reduce `BATCH_SIZE` to 2 or 1, increase `GRADIENT_ACCUM` to maintain effective batch size |
| Slow training | Use `--mixed_precision="fp16"` (xformers optional, not required) |
| No samples | Run `python -m data.dataset` then `python -m data.video_loader` |
| Evaluation pipeline fails | Check that all 4 models are trained and saved in `outputs/` directory |


## Citation

Based on [diffusers examples](https://github.com/huggingface/diffusers/tree/main/examples) and [Something-Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something) dataset.
