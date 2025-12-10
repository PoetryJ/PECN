import os
HF_MIRROR = "https://hf-mirror.com"
# 设置镜像
os.environ["HF_ENDPOINT"] = HF_MIRROR
import json
import random
import argparse
import torch
from pathlib import Path
from PIL import Image
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler
)



def parse_args():
    parser = argparse.ArgumentParser(description="Sample from InstructPix2Pix model")
    
    # Model and data paths
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/instruct_pix2pix_128",
        help="Directory containing the trained InstructPix2Pix model",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="timbrooks/instruct-pix2pix",
        help="Pretrained model name or path for fallback",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="data/sthv2/frames_128x128",
        help="Directory containing video frames",
    )
    parser.add_argument(
        "--val_annotations",
        type=str,
        default="data/sthv2/annotations/val_filtered.json",
        help="Path to validation annotations JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_results_pix2pix",
        help="Directory to save generated samples",
    )
    
    # Frame indices
    parser.add_argument(
        "--input_frame_idx",
        type=int,
        default=20,
        help="Frame index for input image (default: 20)",
    )
    parser.add_argument(
        "--target_frame_idx",
        type=int,
        default=21,
        help="Frame index for target/ground truth image (default: 21)",
    )
    
    # Sampling parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of diffusion inference steps",
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=1.5,
        help="Image guidance scale for InstructPix2Pix",
    )
    
    # Progress estimator parameters
    parser.add_argument(
        "--use_progress_estimator",
        action="store_true",
        help="Use progress estimator to predict progress and add to prompt",
    )
    parser.add_argument(
        "--progress_model_path",
        type=str,
        default=None,
        help="Path to progress estimator model checkpoint",
    )
    parser.add_argument(
        "--progress_num_frames",
        type=int,
        default=20,
        help="Number of frames to use for progress estimation (default: 20)",
    )
    parser.add_argument(
        "--progress_hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension for progress estimator (default: 512)",
    )
    
    return parser.parse_args()

def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def main():
    args = parse_args()
    
    # Convert paths to Path objects
    frames_dir = Path(args.frames_dir)
    val_annotations = Path(args.val_annotations)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running InstructPix2Pix Sampling (Seed: {args.seed}) ===")
    print(f"Model directory: {model_dir}")
    print(f"Frames directory: {frames_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Input frame index: {args.input_frame_idx}")
    print(f"Target frame index: {args.target_frame_idx}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Use progress estimator: {args.use_progress_estimator}")
    
    setup_seed(args.seed)

    # 2. 加载数据并采样
    if not val_annotations.exists():
        raise FileNotFoundError(f"Validation annotations not found at {val_annotations}")
    
    with open(val_annotations, 'r') as f:
        val_data = json.load(f)
    
    # 过滤出有效的样本（input image 和 target image 存在的）
    valid_samples = []
    for ann in val_data:
        video_id = ann['id']
        input_path = frames_dir / f"{video_id}_frame_{args.input_frame_idx:05d}.png"
        target_path = frames_dir / f"{video_id}_frame_{args.target_frame_idx:05d}.png"
        if input_path.exists() and target_path.exists():
            valid_samples.append(ann)
    
    if len(valid_samples) == 0:
        raise FileNotFoundError(f"No valid samples found in {val_annotations}. Please check if frames are extracted correctly.")
    
    if len(valid_samples) < args.num_samples:
        print(f"Warning: Only {len(valid_samples)} valid samples found, but {args.num_samples} requested.")
        print(f"Will sample {len(valid_samples)} samples instead.")
        num_samples_to_use = len(valid_samples)
    else:
        num_samples_to_use = args.num_samples
    
    # 从有效样本中随机采样
    samples = random.sample(valid_samples, num_samples_to_use)
    print(f"\nSelected {num_samples_to_use} samples:")

    # 3. 加载模型
    print(f"\nLoading InstructPix2Pix Pipeline...")
    try:
        # 3.1 优先尝试加载本地保存的完整 pipeline
        # 检查是否有完整的 pipeline（存在 model_index.json 表示是完整 pipeline）
        if (model_dir / "model_index.json").exists():
            print(f"  Loading saved pipeline from {model_dir}...")
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                str(model_dir),
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None,
                local_files_only=True  # 只从本地加载
            )
            print("  Saved pipeline loaded.")
        else:
            # 3.2 Fallback: 从 Hub 加载基础 pipeline，然后替换 UNet
            print(f"  No saved pipeline found. Loading base model from Hub...")
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                args.pretrained_model,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None
            )
            
            # 3.3 加载微调后的 UNet
            final_unet_path = model_dir / "unet"
            if final_unet_path.exists():
                print(f"  Loading fine-tuned UNet from {final_unet_path}...")
                trained_unet = UNet2DConditionModel.from_pretrained(
                    final_unet_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
                pipeline.unet = trained_unet
                print("  Fine-tuned UNet loaded.")
            else:
                print(f"  Warning: Fine-tuned UNet not found. Using base model.")

        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        pipeline.to(device)
        pipeline.enable_model_cpu_offload()
        print("  Pipeline ready.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check your path or network connection.")
        return

    # 3.5. 加载进度估计器（如果启用）
    progress_estimator = None
    if args.use_progress_estimator:
        if args.progress_model_path is None:
            raise ValueError("--progress_model_path must be provided when --use_progress_estimator is enabled")
        
        print(f"\nLoading Progress Estimator...")
        try:
            # 导入进度估计器模块
            import sys
            import importlib.util
            
            progress_evaluator_path = Path(__file__).parent / "progress_evaluator"
            if not progress_evaluator_path.exists():
                raise ImportError(f"Progress evaluator directory not found at {progress_evaluator_path}")
            
            # 将 progress_evaluator 目录添加到 Python 路径，以便相对导入能正常工作
            sys.path.insert(0, str(progress_evaluator_path))
            
            # 导入模块（动态导入，linter 可能无法识别）
            from predict import CorrectSSv2Evaluator  # type: ignore
            
            progress_estimator = CorrectSSv2Evaluator(
                model_path=args.progress_model_path,
                num_frames=args.progress_num_frames,
                hidden_dim=args.progress_hidden_dim
            )
            print("  Progress estimator loaded successfully.")
        except Exception as e:
            import traceback
            print(f"Error loading progress estimator: {e}")
            print(traceback.format_exc())
            print("Please check your progress model path and dependencies.")
            return

    # 4. 生成
    print("\nStarting generation...")
    output_dir.mkdir(exist_ok=True, parents=True)
    # 创建子目录
    (output_dir / 'target').mkdir(parents=True, exist_ok=True)
    (output_dir / 'output').mkdir(parents=True, exist_ok=True)
    (output_dir / 'comparison').mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples, 1):
        video_id = sample['id']
        prompt = sample['label']
        
        input_path = frames_dir / f"{video_id}_frame_{args.input_frame_idx:05d}.png"
        target_path = frames_dir / f"{video_id}_frame_{args.target_frame_idx:05d}.png"
            
        input_image = load_image(input_path)
        
        # 如果启用进度估计，预测进度并添加到 prompt
        if args.use_progress_estimator and progress_estimator is not None:
            try:
                # 获取视频路径和总帧数
                video_path = sample.get('video_path')
                total_frames = sample.get('num_frames')
                
                if video_path and total_frames:
                    # 确保视频路径是绝对路径
                    if not Path(video_path).is_absolute():
                        video_path = Path(__file__).parent / video_path
                    else:
                        video_path = Path(video_path)
                    
                    # 预测进度
                    result = progress_estimator.evaluate_single_video(
                        video_path=str(video_path),
                        label=sample['label'],
                        total_frames=total_frames
                    )
                    
                    if result and 'pred_percentage' in result:
                        progress = result['pred_percentage']
                        prompt += f" And {progress}% of the action has been completed."
                        print(f"  [{i}/{len(samples)}] Video {video_id}: Predicted progress = {progress}%")
                    else:
                        print(f"  [{i}/{len(samples)}] Warning: Failed to predict progress for video {video_id}, using prompt without progress")
                else:
                    print(f"  [{i}/{len(samples)}] Warning: Missing video_path or num_frames for video {video_id}, using prompt without progress")
            except Exception as e:
                print(f"  [{i}/{len(samples)}] Warning: Progress estimation failed for video {video_id}: {e}, using prompt without progress")
        
        generator = torch.Generator(device=device).manual_seed(args.seed)
        
        # 生成
        output = pipeline(
            prompt,
            image=input_image,
            num_inference_steps=args.num_inference_steps,
            image_guidance_scale=args.image_guidance_scale,
            generator=generator
        ).images[0]
        
        # 保存 Grid: [Input, GT, Output]
        if target_path.exists():
            target_image = load_image(target_path)
            # 创建对比图
            w, h = input_image.size
            grid = Image.new('RGB', (w * 3, h))
            grid.paste(input_image, (0, 0))
            grid.paste(target_image, (w, 0))
            grid.paste(output, (w * 2, 0))
            
            (output_dir / 'target').mkdir(parents=True, exist_ok=True)
            (output_dir / 'output').mkdir(parents=True, exist_ok=True)
            (output_dir / 'comparison').mkdir(parents=True, exist_ok=True)
            target_image.save(output_dir / 'target' / f"{video_id}.png")
            output.save(output_dir / 'output' / f"{video_id}.png")
            save_path = output_dir / 'comparison' / f"{video_id}_comparison.png"
            grid.save(save_path)
            print(f"  [{i}/{len(samples)}] Saved comparison to {save_path}")
        else:
            save_path = output_dir / f"{video_id}_output.png"
            output.save(save_path)
            print(f"  [{i}/{len(samples)}] Saved output to {save_path}")

    print(f"\nDone! Generated {len(samples)} samples. Results saved in {output_dir}")

if __name__ == "__main__":
    main()

