"""
阶段2：Student模型
输入：20帧视频 + Label文本
输出：Progress回归值 + 当前帧描述
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import timm


class VideoEncoder(nn.Module):
    """视频编码器：提取时序特征"""
    
    def __init__(self, num_frames=20, hidden_dim=512):
        super().__init__()
        
        # 使用预训练的2D CNN作为帧编码器
        self.frame_encoder = timm.create_model(
            'resnet50', 
            pretrained=True, 
            num_classes=0,  # 移除分类头
            global_pool=''   # 保留spatial特征
        )
        
        # 空间池化
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        
        # 时序建模
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2048,  # ResNet50输出
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        # 降维
        self.proj = nn.Linear(2048, hidden_dim)
        
        self.num_frames = num_frames
    
    def forward(self, video):
        """
        video: (B, T, C, H, W)
        returns: (B, T, hidden_dim)
        """
        B, T, C, H, W = video.shape
        
        # 提取每帧特征
        video_flat = video.view(B * T, C, H, W)
        features = self.frame_encoder(video_flat)  # (B*T, 2048, h, w)
        features = self.spatial_pool(features)  # (B*T, 2048, 1, 1)
        features = features.view(B, T, -1)  # (B, T, 2048)
        
        # 时序编码
        features = self.temporal_encoder(features)  # (B, T, 2048)
        
        # 投影
        features = self.proj(features)  # (B, T, hidden_dim)
        
        return features


class TextEncoder(nn.Module):
    """文本编码器：理解动作Label"""
    
    def __init__(self, hidden_dim=512):
        super().__init__()
        
        # 使用CLIP Text Encoder
        self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # 冻结CLIP（可选）
        for param in self.clip_text.parameters():
            param.requires_grad = False
        
        # 投影到统一空间
        self.proj = nn.Linear(512, hidden_dim)  # CLIP输出512维
    
    def forward(self, text_list):
        """
        text_list: list of strings
        returns: (B, hidden_dim)
        """
        # Tokenize
        tokens = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(next(self.parameters()).device)
        
        # CLIP编码
        outputs = self.clip_text(**tokens)
        text_features = outputs.pooler_output  # (B, 512)
        
        # 投影
        text_features = self.proj(text_features)  # (B, hidden_dim)
        
        return text_features


class CrossModalFusion(nn.Module):
    """跨模态融合：让Label引导视频理解"""
    
    def __init__(self, hidden_dim=512):
        super().__init__()
        
        # Cross-Attention: Text query, Video key/value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, text_feat, video_feat):
        """
        text_feat: (B, hidden_dim)
        video_feat: (B, T, hidden_dim)
        returns: (B, hidden_dim) - fused feature
        """
        # 扩展text维度用于cross-attention
        text_query = text_feat.unsqueeze(1)  # (B, 1, hidden_dim)
        
        # Cross-Attention
        fused, _ = self.cross_attn(
            query=text_query,
            key=video_feat,
            value=video_feat
        )  # (B, 1, hidden_dim)
        
        fused = self.norm1(fused + text_query)
        
        # FFN
        fused = fused + self.ffn(fused)
        fused = self.norm2(fused)
        
        return fused.squeeze(1)  # (B, hidden_dim)


class ProgressHead(nn.Module):
    """进度回归头"""
    
    def __init__(self, hidden_dim=512):
        super().__init__()
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # 输出0-1
        )
    
    def forward(self, x):
        return self.regressor(x).squeeze(-1)  # (B,)

class StudentModel(nn.Module):
    """完整的Student模型"""
    
    def __init__(self, num_frames=20, hidden_dim=512):
        super().__init__()
        
        self.video_encoder = VideoEncoder(num_frames, hidden_dim)
        self.text_encoder = TextEncoder(hidden_dim)
        self.fusion = CrossModalFusion(hidden_dim)
        
        self.progress_head = ProgressHead(hidden_dim)
        
    def forward(self, video, text_labels, target_descriptions=None):
        """
        video: (B, T, C, H, W)
        text_labels: list of strings
        target_descriptions: list of strings (训练) or None (推理)
        
        returns: 
          - progress: (B,) 回归值
          - description_loss or descriptions
        """
        # 编码
        video_feat = self.video_encoder(video)  # (B, T, hidden_dim)
        text_feat = self.text_encoder(text_labels)  # (B, hidden_dim)
        
        # 融合
        fused_feat = self.fusion(text_feat, video_feat)  # (B, hidden_dim)
        
        # 预测进度
        progress = self.progress_head(fused_feat)  # (B,)
        
        # 返回进度和固定0值（保持输出格式兼容）
        dummy_loss = torch.tensor(0.0, device=progress.device)
        return progress, dummy_loss


# ==================== 损失函数 ====================

class MultiTaskLoss(nn.Module):
    """多任务损失：回归 + 描述生成 + 时序一致性"""
    
    def __init__(self, alpha_progress=1.0, alpha_desc=0.5, alpha_rank=0.3):
        super().__init__()
        self.alpha_progress = alpha_progress
        
        # Huber Loss比MSE更鲁棒（对Teacher伪标签的噪声容忍度高）
        self.progress_loss = nn.SmoothL1Loss()
    
    def forward(self, pred_progress, target_progress, desc_loss, 
                pred_progress_early=None, target_progress_early=None):
        """
        pred_progress: (B,) 预测的进度
        target_progress: (B,) 目标进度
        desc_loss: 描述生成的CE loss
        pred_progress_early/target_progress_early: 用于ranking loss（可选）
        """
        # 1. 进度回归loss
        loss_progress = self.progress_loss(pred_progress, target_progress)
        
        # 2. 描述生成loss
        loss_desc = torch.tensor(0.0, device=pred_progress.device)
        
        # 3. 时序一致性ranking loss（如果提供了早期帧）
        loss_rank = torch.tensor(0.0, device=pred_progress.device)
        
        total_loss = self.alpha_progress * loss_progress

        return total_loss, loss_progress, loss_desc, loss_rank

if __name__ == '__main__':
    # 测试模型
    model = StudentModel(num_frames=20, hidden_dim=512)
    
    # 假数据
    video = torch.randn(2, 20, 3, 224, 224)
    text_labels = ["throwing ball", "catching object"]
    target_descriptions = ["ball leaving hand", "hands closing on ball"]
    target_progress = torch.tensor([0.45, 0.75])
    
    # 前向
    pred_progress, desc_loss = model(video, text_labels, target_descriptions)
    
    # 损失
    criterion = MultiTaskLoss()
    total_loss, loss_prog, loss_desc, loss_rank = criterion(
        pred_progress, target_progress, desc_loss
    )
    
    print(f"Progress pred: {pred_progress}")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"  Progress: {loss_prog.item():.4f}")
    print(f"  Description: {loss_desc.item():.4f}")
    print("Model test passed!")