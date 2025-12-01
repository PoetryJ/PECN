import os
import cv2
import numpy as np
import json
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SSIM and PSNR")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="test_result/instruct_pix2pix_128_progress",
        help="Directory to save target samples",
    )
    return parser.parse_args()


def calculate_ssim_psnr(target_img, output_img):
    target_img = cv2.imread(target_img)
    output_img = cv2.imread(output_img)

    # 确保图像尺寸相同
    if target_img.shape != output_img.shape:
        target_img = cv2.resize(target_img, (output_img.shape[1], output_img.shape[0]))

    # 转换为灰度图像计算SSIM
    target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    output_img_gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    # 计算SSIM
    ssim_value = ssim(target_img_gray, output_img_gray, data_range=output_img_gray.max() - output_img_gray.min())

    # 计算PSNR
    psnr_value = psnr(target_img, output_img)

    return ssim_value, psnr_value

def batch_test_ssim_psnr(data_dir):
    results = []

    target_dir = os.path.join(data_dir, 'target')
    output_dir = os.path.join(data_dir, 'output')

    # 获取图像文件列表
    target_imgs = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    output_imgs = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"找到 {len(target_imgs)} 个原始图像")
    print(f"找到 {len(output_imgs)} 个处理图像")

    for target_img in target_imgs:
        if target_img in output_imgs:
            target_path = os.path.join(target_dir, target_img)
            output_path = os.path.join(output_dir, target_img)

            try:
                ssim_val, psnr_val = calculate_ssim_psnr(target_path, output_path)
                results.append({
                    'filename': target_img,
                    'ssim': ssim_val,
                    'psnr': psnr_val
                })
            except Exception as e:
                print(f"处理 {target_img} 时出错: {e}")

    # 计算平均值
    if results:
        avg_ssim = np.mean([r['ssim'] for r in results])
        avg_psnr = np.mean([r['psnr'] for r in results])
        results.append({
            'filename': 'avg_value',
            'ssim': avg_ssim,
            'psnr': avg_psnr
        })

        print(f"平均 SSIM: {avg_ssim:.4f}")
        print(f"平均 PSNR: {avg_psnr:.2f} dB")
        print(f"测试图像数量: {len(results)-1}")


    # 保存结果
    with open(data_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    batch_test_ssim_psnr(data_dir)

if __name__ == "__main__":
    main()

