import os
HF_MIRROR = "https://hf-mirror.com"
# 设置镜像
os.environ["HF_ENDPOINT"] = HF_MIRROR
import json
import random
import torch
from pathlib import Path
from PIL import Image
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler
)

# 1. 配置
SEED = 42
NUM_SAMPLES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 路径配置
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data" / "sthv2"
FRAMES_DIR = DATA_DIR / "frames_128x128"
VAL_ANNOTATIONS = DATA_DIR / "annotations" / "val_filtered.json"
INSTRUCT_PIX2PIX_MODEL_DIR = BASE_DIR / "outputs" / "instruct_pix2pix_128"
PRETRAINED_MODEL = "timbrooks/instruct-pix2pix"



def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def main():
    print(f"=== Running InstructPix2Pix Sampling (Seed: {SEED}) ===")
    setup_seed(SEED)

    # 2. 加载数据并采样
    if not VAL_ANNOTATIONS.exists():
        raise FileNotFoundError(f"Validation annotations not found at {VAL_ANNOTATIONS}")
    
    with open(VAL_ANNOTATIONS, 'r') as f:
        val_data = json.load(f)
    
    # 随机采样
    samples = random.sample(val_data, NUM_SAMPLES)
    print(f"Selected {NUM_SAMPLES} samples:")
    for s in samples:
        print(f"  - {s['id']}: {s['label']}")

    # 3. 加载模型
    print(f"\nLoading InstructPix2Pix Pipeline...")
    try:
        # 3.1 加载基础 Pipeline
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            PRETRAINED_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None
        )
        
        # 3.2 加载微调后的 UNet
        # 优先尝试加载最终模型 (outputs/instruct_pix2pix_128/unet)
        final_unet_path = INSTRUCT_PIX2PIX_MODEL_DIR / "unet"
        
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
            print(f"  Warning: Fine-tuned UNet not found at {final_unet_path}. Using base model.")

        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        pipeline.to(DEVICE)
        pipeline.enable_model_cpu_offload()
        print("  Pipeline ready.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check your path or network connection.")
        return

    # 4. 生成
    print("\nStarting generation...")
    output_dir = Path("test_results_pix2pix")
    output_dir.mkdir(exist_ok=True)

    for i, sample in enumerate(samples, 1):
        video_id = sample['id']
        prompt = sample['label']
        
        input_path = FRAMES_DIR / f"{video_id}_frame_00020.png"
        target_path = FRAMES_DIR / f"{video_id}_frame_00021.png"
        
        if not input_path.exists():
            print(f"  Warning: Input image not found: {input_path}")
            continue
            
        input_image = load_image(input_path)
        
        generator = torch.Generator(device=DEVICE).manual_seed(SEED)
        
        # 生成
        output = pipeline(
            prompt,
            image=input_image,
            num_inference_steps=20,
            image_guidance_scale=1.5,
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
            grid.save(save_path)
            print(f"  [{i}/{NUM_SAMPLES}] Saved comparison to {save_path}")
        else:
            save_path = output_dir / f"{video_id}_output.png"
            output.save(save_path)
            print(f"  [{i}/{NUM_SAMPLES}] Saved output to {save_path}")

    print(f"\nDone! Results saved in {output_dir}")

if __name__ == "__main__":
    main()

