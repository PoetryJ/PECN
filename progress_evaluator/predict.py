"""
正确的SSv2评估脚本：用前20帧预测第20帧的进度
"""

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import csv

from train_reg import get_transforms
from student import StudentModel


class CorrectSSv2Evaluator:
    """正确的SSv2评估器：前20帧预测第20帧进度"""
    
    def __init__(self, model_path, num_frames=20, hidden_dim=512):
        """
        初始化评估器
        
        Args:
            model_path: 模型checkpoint路径
            num_frames: 帧数（必须是20）
            hidden_dim: 隐藏维度（与训练时一致）
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        # 加载模型
        self.model = StudentModel(
            num_frames=num_frames,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        self.num_frames = num_frames
        self.transform = get_transforms(is_train=False)
        
        print(f"Model loaded successfully: {model_path}")
        print(f"  Frames: {num_frames}")
        print(f"  Hidden dimension: {hidden_dim}")
        print(f"  Task: Predict progress at frame {num_frames} using first {num_frames} frames")
        
        # 统计信息
        self.results = []
        self.metrics = defaultdict(list)
    
    def load_first_n_frames(self, video_path, n_frames):
        """
        加载视频的前n帧
        
        Args:
            video_path: 视频路径
            n_frames: 要加载的帧数
            
        Returns:
            frames: 加载的帧列表
            success: 是否成功加载
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 如果视频总帧数不足n_frames，跳过
        if total_frames < n_frames:
            cap.release()
            return None, False
        
        frames = []
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                # 如果中间读取失败，用最后一帧填充
                if frames:
                    frame = frames[-1]
                else:
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            frames.append(frame)
        
        cap.release()
        return frames, True
    
    def evaluate_single_video(self, video_path, label, total_frames):
        """
        评估单个视频
        
        Args:
            video_path: 视频路径
            label: 动作标签
            total_frames: 总帧数
            
        Returns:
            result: 预测结果字典
        """
        # 计算真实进度：第20帧的进度 = 20 / 总帧数
        target_progress = self.num_frames / total_frames if total_frames > 0 else 0
        
        # 如果视频太短，跳过
        if total_frames < self.num_frames:
            return None
        
        # 加载前20帧
        frames, success = self.load_first_n_frames(video_path, self.num_frames)
        if not success:
            return None
        
        # 转换为张量
        video_tensor = torch.stack(frames).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        
        # 预测
        with torch.no_grad():
            pred_progress, _ = self.model(video_tensor, [label])
            pred_progress = pred_progress[0].item()
        
        # 计算误差
        mae = abs(pred_progress - target_progress)
        
        result = {
            'video_id': os.path.basename(video_path).replace('.webm', ''),
            'video_path': video_path,
            'label': label,
            'total_frames': total_frames,
            'frames_used': self.num_frames,
            'pred_progress': pred_progress,
            'target_progress': target_progress,
            'mae': mae,
            'pred_percentage': int(pred_progress * 100),
            'target_percentage': int(target_progress * 100),
            'frame_ratio': f"{self.num_frames}/{total_frames}"
        }
        
        return result
    
    def evaluate_json(self, json_path, output_dir=None, max_videos=None):
        """
        评估JSON文件中的所有视频
        
        Args:
            json_path: JSON文件路径
            output_dir: 输出目录（可选）
            max_videos: 最大视频数（可选，用于测试）
        """
        # 加载JSON数据
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 限制视频数（用于测试）
        if max_videos:
            data = data[:max_videos]
        
        print(f"Starting evaluation of {len(data)} videos...")
        print(f"  Task: Predict progress at frame {self.num_frames} using first {self.num_frames} frames")
        print("="*60)
        
        # 评估每个视频
        self.results = []
        skipped_videos = 0
        
        for video_info in tqdm(data, desc="Evaluating videos"):
            try:
                video_path = video_info['video_path']
                label = video_info['label']
                total_frames = video_info['num_frames']
                
                # 跳过帧数太少的视频
                if total_frames < self.num_frames:
                    skipped_videos += 1
                    continue
                
                # 评估单个视频
                result = self.evaluate_single_video(video_path, label, total_frames)
                
                if result:
                    self.results.append(result)
                    
                    # 收集指标
                    self.metrics['mae'].append(result['mae'])
                    self.metrics['pred_progress'].append(result['pred_progress'])
                    self.metrics['target_progress'].append(result['target_progress'])
                
            except Exception as e:
                print(f"\n❌ Video evaluation failed: {e}")
                print(f"   Video: {video_info.get('video_path', 'N/A')}")
                continue
        
        # 计算总体指标
        self.calculate_metrics()
        
        # 输出结果
        self.print_results(skipped_videos)
        
        # 保存结果
        if output_dir:
            self.save_results(output_dir)
        
        return self.results
    
    def calculate_metrics(self):
        """计算评估指标"""
        if not self.results:
            return
        
        mae_values = np.array(self.metrics['mae']) * 100  # 转换为百分比误差
        
        # 计算三个准确率等级
        very_accurate = np.sum(mae_values <= 2) / len(mae_values) * 100
        nearly_accurate = np.sum(mae_values <= 8) / len(mae_values) * 100
        reasonably_accurate = np.sum(mae_values <= 15) / len(mae_values) * 100
        
        self.overall_metrics = {
            'num_videos': len(self.results),
            'mae_mean': np.mean(mae_values),
            'very_accurate_percentage': very_accurate,
            'nearly_accurate_percentage': nearly_accurate,
            'reasonably_accurate_percentage': reasonably_accurate,
            'accuracy_distribution': {
                'very_accurate_count': int(np.sum(mae_values <= 2)),
                'nearly_accurate_count': int(np.sum(mae_values <= 8)),
                'reasonably_accurate_count': int(np.sum(mae_values <= 15)),
                'total_count': len(mae_values)
            }
        }
        
        # 为图表准备数据
        self.accuracy_counts = [
            int(np.sum(mae_values <= 2)),
            int(np.sum((mae_values > 2) & (mae_values <= 8))),
            int(np.sum((mae_values > 8) & (mae_values <= 15))),
            int(np.sum(mae_values > 15))
        ]
        
        self.accuracy_labels = ['Very Accurate (≤2%)', 'Nearly Accurate (≤8%)', 
                               'Reasonably Accurate (≤15%)', 'Inaccurate (>15%)']
       
    def print_results(self, skipped_videos):
        """打印评估结果"""
        if not hasattr(self, 'overall_metrics'):
            self.calculate_metrics()
        
        print("\n" + "="*60)
        print("SSv2 Evaluation Results")
        print("="*60)
        
        metrics = self.overall_metrics
        print(f"Valid videos: {metrics['num_videos']}")
        print(f"Skipped videos (too short): {skipped_videos}")
        print(f"\nMAE: {metrics['mae_mean']:.2f}%")
        print(f"\nAccuracy Metrics:")
        print(f"  Very Accurate (≤2% error): {metrics['very_accurate_percentage']:.2f}%")
        print(f"  Nearly Accurate (≤8% error): {metrics['nearly_accurate_percentage']:.2f}%")
        print(f"  Reasonably Accurate (≤15% error): {metrics['reasonably_accurate_percentage']:.2f}%")
        
        # 显示前10个样本的详细结果
        print(f"\n🔍 First 10 sample results:")
        print(f"{'#':<4} {'Video ID':<12} {'Frames':<8} {'Pred%':<8} {'True%':<8} {'MAE%':<8} {'Frame Ratio'}")
        print("-" * 70)
        for i, result in enumerate(self.results[:10]):
            print(f"{i+1:<4} {result['video_id']:<12} {result['total_frames']:<8.0f} "
                  f"{int(result['pred_percentage']):<8} {int(result['target_percentage']):<8} "
                  f"{result['mae']*100:<8.1f} {result['frame_ratio']}")
        
       
    def save_results(self, output_dir):
        """保存结果到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存指标到JSON
        json_path = os.path.join(output_dir, 'evaluation_metrics.json')
        with open(json_path, 'w') as f:
            json.dump({
                'overall_metrics': self.overall_metrics,
                'accuracy_distribution': self.overall_metrics['accuracy_distribution']
            }, f, indent=2)
        
        # 2. 生成可视化图表
        self.plot_results(output_dir)
        
        print(f"\n💾 Results saved to: {output_dir}")
        print(f"  Evaluation metrics: {json_path}")
    
    def plot_results(self, output_dir):
        """生成可视化图表"""
        try:
            # 1. 散点图：预测vs真实
            plt.figure(figsize=(10, 8))
            targets = np.array(self.metrics['target_progress']) * 100
            preds = np.array(self.metrics['pred_progress']) * 100
            
            plt.scatter(targets, preds, alpha=0.5, s=20)
            plt.plot([0, 100], [0, 100], 'r--', label='Ideal Prediction', alpha=0.7)
            plt.xlabel('True Progress (20/Total Frames) (%)')
            plt.ylabel('Predicted Progress (%)')
            plt.title(f'Frame {self.num_frames} Progress Prediction (MAE={self.overall_metrics["mae_mean"]:.2f}%)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'scatter_plot.png'), dpi=150)
            plt.close()
            
            # 2. 误差分布直方图
            plt.figure(figsize=(10, 6))
            errors = np.array(self.metrics['mae']) * 100
            plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel('Absolute Error (%)')
            plt.ylabel('Number of Samples')
            plt.title(f'Error Distribution (Mean={errors.mean():.2f}%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=150)
            plt.close()
            
            # 3. 准确率分布图
            plt.figure(figsize=(10, 7))
            
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # 左边：饼图
            colors = ['#4CAF50', '#8BC34A', '#FFC107', '#F44336']
            wedges, texts, autotexts = ax1.pie(
                self.accuracy_counts, 
                labels=self.accuracy_labels, 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            
            # 美化饼图文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax1.set_title('Accuracy Distribution by Error Threshold')
            
            # 右边：条形图
            percentages = [
                self.overall_metrics['very_accurate_percentage'],
                self.overall_metrics['nearly_accurate_percentage'] - self.overall_metrics['very_accurate_percentage'],
                self.overall_metrics['reasonably_accurate_percentage'] - self.overall_metrics['nearly_accurate_percentage'],
                100 - self.overall_metrics['reasonably_accurate_percentage']
            ]
            
            bars = ax2.bar(self.accuracy_labels, percentages, color=colors, alpha=0.8)
            ax2.set_ylabel('Percentage of Videos (%)')
            ax2.set_title('Cumulative Accuracy by Error Threshold')
            ax2.set_ylim(0, 100)
            ax2.grid(True, axis='y', alpha=0.3)
            
            # 在条形图上添加数值标签
            for bar, percentage in zip(bars, percentages):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'accuracy_distribution.png'), dpi=150)
            plt.close()
            
            print(f"📊 Visualizations saved: scatter_plot.png, error_distribution.png, accuracy_distribution.png")
            
        except Exception as e:
            print(f"⚠️  Error generating charts: {e}")


def main():
    parser = argparse.ArgumentParser(description='SSv2 Evaluation: Predict progress at frame 20 using first 20 frames')
    
    # 必需参数
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--test_json', required=True, help='Path to SSv2 format test JSON file')
    
    # 模型参数（必须与训练时一致）
    parser.add_argument('--num_frames', type=int, default=20, help='Number of frames to use')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    
    # 评估参数
    parser.add_argument('--max_videos', type=int, default=None,
                       help='Maximum number of test videos (for quick testing)')
    
    # 输出选项
    parser.add_argument('--output_dir', type=str, default='./correct_ssv2_evaluation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = CorrectSSv2Evaluator(
        model_path=args.model_path,
        num_frames=args.num_frames,
        hidden_dim=args.hidden_dim
    )
    
    # 运行评估
    results = evaluator.evaluate_json(
        json_path=args.test_json,
        output_dir=args.output_dir,
        max_videos=args.max_videos
    )
    
    # 生成最终报告
    report_path = os.path.join(args.output_dir, 'summary.txt')
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SSv2 Evaluation Summary Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Task: Predict progress at frame {args.num_frames} using first {args.num_frames} frames\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test data: {args.test_json}\n")
        f.write(f"Valid videos: {evaluator.overall_metrics['num_videos']}\n")
        f.write(f"\nPerformance Metrics:\n")
        f.write(f"  MAE: {evaluator.overall_metrics['mae_mean']:.2f}%\n")
        f.write(f"  Very Accurate (≤2% error): {evaluator.overall_metrics['very_accurate_percentage']:.2f}%\n")
        f.write(f"  Nearly Accurate (≤8% error): {evaluator.overall_metrics['nearly_accurate_percentage']:.2f}%\n")
        f.write(f"  Reasonably Accurate (≤15% error): {evaluator.overall_metrics['reasonably_accurate_percentage']:.2f}%\n")
    
    print(f"\n📄 Summary report saved: {report_path}")
    print("="*60)
    print("✅ Evaluation completed!")


if __name__ == '__main__':
    main()