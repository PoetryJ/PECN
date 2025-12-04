"""
阶段2：训练Student模型
使用Teacher生成的伪标签进行训练
"""
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import os
import math
from student import StudentModel, MultiTaskLoss


# train.py 中的数据集类修改
class ProgressDatasetWithPseudoLabels(Dataset):
    """直接使用原始SSv2数据，不使用伪标签"""
    
    def __init__(self, original_json_path, num_frames=20, transform=None):
        with open(original_json_path, encoding='utf-8') as f:
            self.data = json.load(f)  # 直接加载原始数据
        
        self.num_frames = num_frames
        self.transform = transform
        
        # 不展开为样本，而是在训练时动态采样
        print(f"Loaded {len(self.data)} videos")
    
    def _sample_frames_uniformly(self, video_path, total_frames, end_frame):
        """
        从视频[0, end_frame]均匀采样num_frames帧
        """
        cap = cv2.VideoCapture(video_path)
        
        # 计算采样索引
        if end_frame < self.num_frames:
            # 不够，重复帧
            indices = np.linspace(0, end_frame, self.num_frames, dtype=int)
        else:
            indices = np.linspace(0, end_frame, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
            else:
                frame = Image.new('RGB', (224, 224))
            
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        
        cap.release()
        
        return torch.stack(frames)  # (T, C, H, W)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_info = self.data[idx]
        
        # 随机选择查看多少进度（0.1到0.9之间）
        progress_ratio = np.random.uniform(0.1, 0.9)
        end_frame = int(video_info['num_frames'] * progress_ratio)
        
        # 确保至少有一帧
        end_frame = max(1, end_frame)
        
        # 加载视频片段
        frames = self._sample_frames_uniformly(
            video_info['video_path'],
            video_info['num_frames'],
            end_frame
        )
        
        # 获取动作标签
        label = video_info['label']
        
        # 使用物理时间作为ground truth
        semantic_progress = progress_ratio
        
        # 生成简单的描述（固定模板）
        # 这里可以根据progress_ratio生成不同阶段的描述
        if progress_ratio < 0.3:
            description = f"Starting to {label.lower()}"
        elif progress_ratio < 0.7:
            description = f"In the middle of {label.lower()}"
        else:
            description = f"Finishing {label.lower()}"
        
        return {
            'video': frames,  # (T, C, H, W)
            'label': label,
            'description': description,
            'progress': torch.tensor(semantic_progress, dtype=torch.float32)
        }


def get_transforms(is_train=True):
    """数据增强"""
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    if is_train:
        return T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            normalize
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])


def train_student(args):
    """训练Student模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 数据
    train_dataset = ProgressDatasetWithPseudoLabels(
        args.train_data_json,  # 改为args.train_data_json
        num_frames=args.num_frames,
        transform=get_transforms(is_train=True)
    )
    
    val_dataset = ProgressDatasetWithPseudoLabels(
        args.val_data_json,  # 改为args.val_data_json
        num_frames=args.num_frames,
        transform=get_transforms(is_train=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # 模型
    model = StudentModel(
        num_frames=args.num_frames,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # 损失和优化器
    criterion = MultiTaskLoss(
        alpha_progress=1.0,
        alpha_desc=0.0,
        alpha_rank=0.0
    )
    
    # 分层学习率
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.2},
        {'params': head_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # 训练循环
    best_mae = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_mae = 0
        train_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            videos = batch['video'].cuda()  # (B, T, C, H, W)
            labels = batch['label']
            descriptions = batch['description']
            target_progress = batch['progress'].cuda()
            
            optimizer.zero_grad()
            
            # 前向
            pred_progress, desc_loss = model(videos, labels, descriptions)
            
            # 损失
            total_loss, loss_prog, loss_desc, loss_rank = criterion(
                pred_progress, target_progress, desc_loss
            )
            
            # 反向
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 统计
            mae = (pred_progress - target_progress).abs().mean()
            train_loss += total_loss.item() * videos.size(0)
            train_mae += mae.item() * videos.size(0)
            train_samples += videos.size(0)
            
            pbar.set_postfix({
                'loss': f'{train_loss/train_samples:.4f}',
                'MAE': f'{train_mae/train_samples*100:.1f}%'
            })
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        val_mae = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                videos = batch['video'].cuda()
                labels = batch['label']
                descriptions = batch['description']
                target_progress = batch['progress'].cuda()
                
                pred_progress, desc_loss = model(videos, labels, descriptions)
                
                total_loss, loss_prog, loss_desc, loss_rank = criterion(
                    pred_progress, target_progress, desc_loss
                )
                
                mae = (pred_progress - target_progress).abs().mean()
                val_loss += total_loss.item() * videos.size(0)
                val_mae += mae.item() * videos.size(0)
                val_samples += videos.size(0)
        
        avg_train_mae = train_mae / train_samples * 100
        avg_val_mae = val_mae / val_samples * 100
        
        print(f'\nEpoch {epoch+1}:')
        print(f'  Train MAE: {avg_train_mae:.1f}%')
        print(f'  Val MAE:   {avg_val_mae:.1f}%')
        
        # 保存最佳模型
        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'mae': avg_val_mae
            }, os.path.join(args.save_dir, 'best_student.pth'))
            print(f'  ✅ Saved best model (MAE: {avg_val_mae:.1f}%)')
        
        # 保存checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(args.save_dir, f'checkpoint_ep{epoch+1}.pth'))
    
    print(f'\n🎉 Training complete! Best Val MAE: {best_mae:.1f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 修改参数名称，表示这是原始数据而非伪标签数据
    parser.add_argument('--train_data_json', required=True,
                        help='Training data (original SSv2 format)')
    parser.add_argument('--val_data_json', required=True,
                        help='Validation data (original SSv2 format)')
    
    parser.add_argument('--num_frames', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=512)
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
 
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 修改函数调用参数名
    train_student(args)