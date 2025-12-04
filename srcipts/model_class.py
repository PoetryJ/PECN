"""阶段2：Student模型（分类版本）
输入：20帧视频 + Label文本
输出：Progress分类（0-100%） + 进度值（用于回归对比）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
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


class ClassificationHead(nn.Module):
    """进度分类头（将0-100%分成多个类别）"""
    
    def __init__(self, hidden_dim=512, num_classes=101):
        super().__init__()
        
        self.num_classes = num_classes
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def forward(self, x):
        """
        x: (B, hidden_dim)
        returns:
          - logits: (B, num_classes) 分类logits
          - progress: (B,) 转换为0-1之间的进度值（用于输出）
        """
        logits = self.classifier(x)  # (B, num_classes)
        
        # 将分类结果转换为连续的进度值（可选，用于输出）
        # 使用softmax获取每个类别的概率
        probs = F.softmax(logits, dim=-1)
        
        # 计算期望进度值（0-1之间）
        # 创建一个从0到1的线性空间，代表每个类别的中心值
        class_centers = torch.linspace(0, 1, self.num_classes, device=x.device)
        
        # 计算期望值（概率加权平均）
        progress = torch.sum(probs * class_centers, dim=-1)  # (B,)
        
        return logits, progress


class StudentModel(nn.Module):
    """简化的Student模型（分类版本）"""
    
    def __init__(self, num_frames=20, hidden_dim=512, num_classes=101):
        super().__init__()
        
        self.video_encoder = VideoEncoder(num_frames, hidden_dim)
        self.text_encoder = TextEncoder(hidden_dim)
        self.fusion = CrossModalFusion(hidden_dim)
        
        # 使用分类头
        self.classification_head = ClassificationHead(hidden_dim, num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, video, text_labels, target_descriptions=None):
        """
        video: (B, T, C, H, W)
        text_labels: list of strings
        target_descriptions: 已移除，保持参数兼容性
        
        returns: 
          - logits: (B, num_classes) 分类logits
          - progress: (B,) 转换后的进度值（0-1）
          - dummy_loss: 固定为0，保持输出格式兼容
        """
        # 编码
        video_feat = self.video_encoder(video)  # (B, T, hidden_dim)
        text_feat = self.text_encoder(text_labels)  # (B, hidden_dim)
        
        # 融合
        fused_feat = self.fusion(text_feat, video_feat)  # (B, hidden_dim)
        
        # 预测进度（分类）
        logits, progress = self.classification_head(fused_feat)
        
        # 返回logits、进度值和固定0值（保持输出格式兼容）
        dummy_loss = torch.tensor(0.0, device=progress.device)
        return logits, progress, dummy_loss


# ==================== 分类损失函数 ====================

class ClassificationLoss(nn.Module):
    """分类损失函数：交叉熵损失"""
    
    def __init__(self, num_classes=101):
        super().__init__()
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, pred_logits, target_progress, desc_loss=None):
        """
        pred_logits: (B, num_classes) 分类logits
        target_progress: (B,) 目标进度值（0-1之间）
        desc_loss: 已移除，保持参数兼容性
        """
        # 将目标进度值（0-1）转换为类别标签（0到num_classes-1）
        # 需要将target_progress从[0, 1]映射到[0, num_classes-1]
        target_labels = (target_progress * (self.num_classes - 1)).long()
        
        # 确保标签在有效范围内
        target_labels = torch.clamp(target_labels, 0, self.num_classes - 1)
        
        # 计算分类损失
        loss_classification = self.cross_entropy(pred_logits, target_labels)
        
        # 保持输出格式兼容
        loss_desc = torch.tensor(0.0, device=pred_logits.device)
        loss_rank = torch.tensor(0.0, device=pred_logits.device)
        
        total_loss = loss_classification
        
        return total_loss, loss_classification, loss_desc, loss_rank


if __name__ == '__main__':
    # 测试模型
    model = StudentModel(num_frames=20, hidden_dim=512, num_classes=101)
    
    # 假数据
    video = torch.randn(2, 20, 3, 224, 224)
    text_labels = ["throwing ball", "catching object"]
    target_progress = torch.tensor([0.45, 0.75])  # 0-1之间的进度值
    
    # 前向
    logits, progress, dummy_loss = model(video, text_labels)
    
    # 损失
    criterion = ClassificationLoss(num_classes=101)
    total_loss, loss_cls, loss_desc, loss_rank = criterion(
        logits, target_progress, dummy_loss
    )
    
    print(f"Logits形状: {logits.shape}")
    print(f"预测进度: {progress}")
    print(f"分类损失: {loss_cls.item():.4f}")
    print("✅ 分类模型测试通过!")