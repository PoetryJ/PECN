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
    
    setup_seed(args.seed)

    # 2. 加载数据并采样
    if not val_annotations.exists():
        raise FileNotFoundError(f"Validation annotations not found at {val_annotations}")
    
    with open(val_annotations, 'r') as f:
        val_data = json.load(f)
    
    # 随机采样
    samples = random.sample(val_data, args.num_samples)
    print(f"\nSelected {args.num_samples} samples:")
    for s in samples:
        print(f"  - {s['id']}: {s['label']}")

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

    # 4. 生成
    print("\nStarting generation...")
    output_dir.mkdir(exist_ok=True, parents=True)

    for i, sample in enumerate(samples, 1):
        video_id = sample['id']
        prompt = sample['label']
        
        input_path = frames_dir / f"{video_id}_frame_{args.input_frame_idx:05d}.png"
        target_path = frames_dir / f"{video_id}_frame_{args.target_frame_idx:05d}.png"
        
        if not input_path.exists():
            print(f"  Warning: Input image not found: {input_path}")
            continue
            
        input_image = load_image(input_path)
        
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
            
            save_path = output_dir / f"{video_id}_comparison.png"
            target_image.save(output_dir / 'target' / f"{video_id}.png")
            output.save(output_dir / 'output' / f"{video_id}.png")
            grid.save(save_path)
            print(f"  [{i}/{args.num_samples}] Saved comparison to {save_path}")
        else:
            save_path = output_dir / f"{video_id}_output.png"
            output.save(save_path)
            print(f"  [{i}/{args.num_samples}] Saved output to {save_path}")

    print(f"\nDone! Results saved in {output_dir}")

if __name__ == "__main__":
    main()

