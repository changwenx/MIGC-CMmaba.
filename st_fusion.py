import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from mamba_vision_model import TrafficMambaVision, Stem, Downsample, ConvBlock, MambaVisionLayer
class SpatioTemporalFusion(nn.Module):
    """修复的五步时空特征融合模块"""
    
    def __init__(self, spatial_dim, temporal_dim, aligned_dim=512, hidden_dim=256):
        super(SpatioTemporalFusion, self).__init__()
        
        # 保存参数作为属性
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.aligned_dim = aligned_dim  # 添加这行
        self.hidden_dim = hidden_dim    # 也可以添加这行
        
        # 1. 通道对齐：将空间和时间特征投影到相同的维度
        self.spatial_align = nn.Linear(spatial_dim, aligned_dim)
        self.temporal_align = nn.Linear(temporal_dim, aligned_dim)
        
        # 2. γ计算：使用BatchNorm获取通道重要性
        self.spatial_norm = nn.BatchNorm1d(aligned_dim, affine=True)
        self.temporal_norm = nn.BatchNorm1d(aligned_dim, affine=True)
        
        # 自适应阈值参数
        self.alpha = 0.1
        
        # 3. 增强：特征增强层
        self.spatial_enhance = nn.Linear(aligned_dim, aligned_dim)
        self.temporal_enhance = nn.Linear(aligned_dim, aligned_dim)
        
        # 4. 拼接后的最终变换
        self.final_fusion = nn.Linear(2 * aligned_dim, temporal_dim)
        
    def get_channel_importance(self, norm_layer):
        """计算通道重要性γ"""
        return torch.abs(norm_layer.weight)
    
    def get_adaptive_threshold(self, gamma):
        """计算自适应阈值"""
        return gamma.min() + self.alpha * (gamma.max() - gamma.min())
    
    def apply_channel_enhancement(self, features, gamma, threshold, enhance_layer):
        """应用通道级增强 - 修复版本"""
        B, N, C = features.shape
        
        # 创建重要性掩码
        important_mask = gamma >= threshold
        
        # 对重要通道：使用增强层
        if important_mask.any():
            # 使用逐元素乘法而不是切片操作
            mask_3d = important_mask.view(1, 1, -1).expand(B, N, -1)
            enhanced_features = torch.where(
                mask_3d,
                enhance_layer(features),  # 对重要通道使用增强层
                features  # 对不重要通道保持原特征
            )
        else:
            enhanced_features = features
        
        return enhanced_features
    
    def forward(self, spatial_feat, temporal_feat):
        """
        五步融合流程：
        1. 通道对齐: (B,N,8) + (B,N,1024) → (B,N,512) × 2
        2. γ计算: → (512,) × 2
        3. 增强: (B,N,512) × 2 → (B,N,512) × 2  
        4. 拼接: (B,N,512) × 2 → (B,N,1024)
        5. 最终输出: (B,N,1024) → (B,N,1024)
        """
        B, N, _ = spatial_feat.shape
        
        # 1. 通道对齐
        spatial_aligned = self.spatial_align(spatial_feat)  # (B,N,8) → (B,N,512)
        temporal_aligned = self.temporal_align(temporal_feat)  # (B,N,1024) → (B,N,512)
        
        # 2. γ计算（通道重要性）
        # 重塑为 (B*N, aligned_dim) 用于BatchNorm
        spatial_flat = spatial_aligned.reshape(-1, self.aligned_dim)
        temporal_flat = temporal_aligned.reshape(-1, self.aligned_dim)
        
        # 通过BatchNorm
        spatial_flat_norm = self.spatial_norm(spatial_flat)
        temporal_flat_norm = self.temporal_norm(temporal_flat)
        
        gamma_spatial = self.get_channel_importance(self.spatial_norm)  # (512,)
        gamma_temporal = self.get_channel_importance(self.temporal_norm)  # (512,)
        
        threshold_spatial = self.get_adaptive_threshold(gamma_spatial)
        threshold_temporal = self.get_adaptive_threshold(gamma_temporal)
        
        # 重塑回原始形状
        spatial_aligned = spatial_flat_norm.reshape(B, N, self.aligned_dim)
        temporal_aligned = temporal_flat_norm.reshape(B, N, self.aligned_dim)
        
        # 3. 增强（使用安全的逐元素操作）
        spatial_enhanced = self.apply_channel_enhancement(
            spatial_aligned, gamma_spatial, threshold_spatial, self.spatial_enhance
        )
        
        temporal_enhanced = self.apply_channel_enhancement(
            temporal_aligned, gamma_temporal, threshold_temporal, self.temporal_enhance
        )
        
        # 4. 拼接
        concatenated = torch.cat([spatial_enhanced, temporal_enhanced], dim=-1)  # (B,N,1024)
        
        # 5. 最终输出
        fused = self.final_fusion(concatenated)  # (B,N,1024)
        
        return fused

import torch
import torch.nn as nn
from dataclasses import dataclass
class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class SelectLastTimestep(nn.Module):
    def forward(self, x):
        return x[:, -1, :]
              
class EnhancedMultiScaleModel(nn.Module):
    def __init__(self, spatial_dim, in_chans=1, num_classes=1, depths=[3, 3, 2, 2],
                 dims=[64, 128, 256, 512, 1024], window_sizes=[8, 8, 14, 7],
                 mlp_ratio=4, drop_rate=0., drop_path_rate=0.2, layer_scale=None,
                 seq_length=12, feature_dim=1):
        super(EnhancedMultiScaleModel, self).__init__()
        
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.spatial_dim = spatial_dim
        
        # 图像特征提取分支
        from mamba_vision_model import TrafficMambaVision
        self.image_backbone = TrafficMambaVision(
            in_chans=in_chans, num_classes=num_classes, depths=depths,
            dims=dims, window_sizes=window_sizes, mlp_ratio=mlp_ratio,
            drop_rate=drop_rate, drop_path_rate=drop_path_rate, layer_scale=layer_scale
        )
        self.image_backbone.head = nn.Identity()
        
        # 时间序列特征提取器
        self.temporal_feature_extractor = nn.Sequential(
            nn.Unflatten(1, (seq_length, feature_dim)),  # (2720, 12, 1)
    
            # 投影到Mamba需要的维度
            nn.Linear(feature_dim, 128),  # (2720, 12, 1) -> (2720, 12, 128)
            nn.ReLU(),
    
            # 转置维度以适应Mamba
            Transpose(0, 1),  # (12, 2720, 128)
    
            # Mamba处理
            Mamba(
                d_model=128,    # 必须与输入维度匹配
                d_state=64,     # 增加状态维度
                d_conv=4,
                expand=4,       # 增加扩展因子
            ),  # (12, 2720, 128)
    
            # 转置回原始维度
            Transpose(0, 1),    # (2720, 12, 128)
    
            # 取最后一个时间步
            SelectLastTimestep(),  # (2720, 128)
    
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
    
            # 最终输出投影到目标维度
            nn.Linear(128, 64)  # (2720, 64)
        )
        
        # 图像特征投影层
        self.image_feature_projection = nn.Sequential(
            nn.Linear(dims[4], 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        
        # 时间序列融合层
        self.temporal_fusion = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128)
        )
        
        # 时间编码嵌入层（调整为2个）
        self.time_of_day_embedding = nn.Embedding(288, 32)
        self.day_of_week_embedding = nn.Embedding(7, 32)
        # 流量特征投影层
        self.flow_feature_projection = nn.Linear(12, 32)
        
        # 时间特征融合层
        self.time_feature_fusion = nn.Sequential(
            nn.Linear(32 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        # 最终时间特征处理器
        self.final_temporal_processor = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # 时空融合模块
        self.st_fusion = SpatioTemporalFusion(
            spatial_dim=spatial_dim,
            temporal_dim=256,
            aligned_dim=256,
            hidden_dim=128
        )
        
        # 预测头
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def extract_flow_features(self, sequences):
        """
        从时间序列中提取原始流量数据
        sequences: (B, T, N, d)
        返回: (B, N, T)
        """
        B, T, N, d = sequences.shape
        print(f"Input sequences shape: {sequences.shape}")
        print(f"d (feature dimension): {d}")
    
        # 提取流量数据（假设第0维是流量）
        flow_data = sequences[..., 0]  # (B, T, N)
        print(f"Flow data shape before transpose: {flow_data.shape}")
    
        # 调整维度：从 (B, T, N) 到 (B, N, T)
        flow_features = flow_data.transpose(1, 2)  # (B, N, T)
        print(f"Flow features shape: {flow_features.shape}")
    
        return flow_features

    def forward(self, x_images, x_sequences, spatial_features, time_info):
        B, N, C, H, W = x_images.shape
        _, T, _, d = x_sequences.shape

        # ============= 图像特征提取 =============
        chunk_size = 1024
        all_features = []
        x_images_flat = x_images.reshape(B * N, C, H, W)

        for start in range(0, B * N, chunk_size):
            end = min(start + chunk_size, B * N)
            batch_chunk = x_images_flat[start:end]
            feats = self.image_backbone(batch_chunk)

            if feats.dim() == 3:
                if feats.shape[1] == 1:
                    feats = feats.squeeze(1)
                else:
                    feats = feats.mean(dim=1)
            all_features.append(feats)

        image_features = torch.cat(all_features, dim=0)
        image_features = image_features.reshape(B, N, -1)
        image_temporal_features = self.image_feature_projection(image_features)  # (B, N, 64)

        # ============= 原始序列特征提取 =============
        sequences_reshaped = x_sequences.permute(0, 2, 1, 3).reshape(B * N, T * d)
        original_temporal_features = self.temporal_feature_extractor(sequences_reshaped)
        original_temporal_features = original_temporal_features.reshape(B, N, -1)  # (B, N, 64)

        # ============= 时间序列特征融合 =============
        combined_temporal = torch.cat([original_temporal_features, image_temporal_features], dim=-1)
        fused_temporal = self.temporal_fusion(combined_temporal)  # (B, N, 128)

        # ============= 时间编码处理 =============
        time_embeddings = []

        # 时间嵌入（小时）
        hours = time_info[..., 0].long()  # (B, T)
        time_embed = self.time_of_day_embedding(hours)  # (B, T, 32)
        time_embeddings.append(time_embed.mean(dim=1).unsqueeze(1).expand(-1, N, -1))  # (B, N, 32)

        # 星期几嵌入
        days = time_info[..., 1].long()  # (B, T)
        day_embed = self.day_of_week_embedding(days)  # (B, T, 32)
        time_embeddings.append(day_embed.mean(dim=1).unsqueeze(1).expand(-1, N, -1))  # (B, N, 32)

        # ============= 流量嵌入 =============
        flow_features = self.extract_flow_features(x_sequences)  # (B, N, 6)
        flow_embed = self.flow_feature_projection(flow_features)  # (B, N, 32)
        time_embeddings.append(flow_embed)

        # ============= 时间特征融合 =============
        time_features = torch.cat(time_embeddings, dim=-1)  # (B, N, 96)
        time_encoded = self.time_feature_fusion(time_features)  # (B, N, 64)

        # ============= 最终时间特征融合 =============
        final_temporal_input = torch.cat([fused_temporal, time_encoded], dim=-1)  # (B, N, 192)
        final_temporal_features = self.final_temporal_processor(final_temporal_input)  # (B, N, 256)

        # ============= 时空融合 =============
        fused_output = self.st_fusion(spatial_features, final_temporal_features)

        # ============= 最终预测 =============
        final_output = self.head(fused_output)

        return final_output, image_features

             
# 在 st_fusion.py 中正确重新定义 EnhancedTrafficMambaVision
class EnhancedTrafficMambaVision(nn.Module):
    def __init__(self, in_chans=1, num_classes=1, depths=[3, 3, 2, 2],
                 dims=[64, 128, 256, 512, 1024], window_sizes=[8, 8, 14, 7],
                 mlp_ratio=4, drop_rate=0., drop_path_rate=0.2, layer_scale=None):
        super(EnhancedTrafficMambaVision, self).__init__()
        
        self.dims = dims
        self.num_classes = num_classes
        
        # Stem
        self.stem = Stem(in_chans=in_chans, out_chans=dims[0])
        
        # Stage 1
        self.stage1 = nn.Sequential(
            *[ConvBlock(dim=dims[0], drop_path=drop_path_rate) for _ in range(depths[0])],
            Downsample(dim=dims[0])
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            *[ConvBlock(dim=dims[1], drop_path=drop_path_rate) for _ in range(depths[1])],
            Downsample(dim=dims[1])
        )
        
        # Stage 3
        self.stage3 = MambaVisionLayer(
            dim=dims[2], depth=depths[2], window_size=window_sizes[2],
            downsample=True, mlp_ratio=mlp_ratio, drop=drop_rate,
            drop_path=drop_path_rate, layer_scale=layer_scale
        )
        
        # Stage 4
        self.stage4 = MambaVisionLayer(
            dim=dims[3], depth=depths[3], window_size=window_sizes[3],
            downsample=True, mlp_ratio=mlp_ratio, drop=drop_rate,
            drop_path=drop_path_rate, layer_scale=layer_scale
        )
        
        self.norm = nn.BatchNorm2d(dims[4])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[4], num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

        
