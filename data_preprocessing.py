from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset

class TrafficDataProcessor:
    """交通数据预处理模块 - 支持NPZ格式"""
    
    def __init__(self, seq_length=12, pred_length=1):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.data_min = None
        self.data_max = None
        self.inn_means = None  # 实例归一化的均值
        self.inn_stds = None   # 实例归一化的标准差
    
    def load_npz_data(self, data_path):
        """加载NPZ格式的数据 - 保持接口不变"""
        data = np.load(data_path)
        
        # 根据不同的NPZ文件结构进行调整
        if 'data' in data:
            # 常见格式: (T, N, F)
            raw_data = data['data']
            # 取流量特征并转置为 (N, T)
            if len(raw_data.shape) == 3:
                raw_data = raw_data[..., 0].T  # 取第一个特征
            else:
                raw_data = raw_data.T
        elif 'x' in data and 'y' in data:
            # 如果已经是处理好的数据
            return data['x'], data['y']
        else:
            # 尝试其他可能的键名
            for key in data.keys():
                if len(data[key].shape) >= 2:
                    raw_data = data[key].T  # 转置为 (N, T)
                    break
            else:
                raise ValueError("无法识别NPZ文件格式")
        
        # 保存原始数据的min/max用于后续处理（保持兼容性）
        self.data_min = np.min(raw_data)
        self.data_max = np.max(raw_data)
        print(f"Data range - Min: {self.data_min:.4f}, Max: {self.data_max:.4f}")
        
        return raw_data
    
    def generate_seq2seq_data(self, data, scaler=None):
        """
        生成序列到序列的数据 - 使用可逆实例归一化（INN）
        :param data: 原始数据 (num_nodes, num_timesteps)
        :return: x, y 数据（已归一化）
        """
        if len(data.shape) != 2:
            raise ValueError("输入数据应该是2维的 (num_nodes, num_timesteps)")
        
        num_nodes, num_timesteps = data.shape
        
        # 应用实例归一化（INN）
        # 计算每个节点的均值和标准差
        means = np.mean(data, axis=1, keepdims=True)
        stds = np.std(data, axis=1, keepdims=True) + 1e-8  # 避免除零
        
        # 标准化数据 (零均值，单位方差)
        data_normalized = (data - means) / stds
        
        print(f"Instance normalized data - Mean: {np.mean(data_normalized):.6f}, Std: {np.std(data_normalized):.6f}")
        
        # 存储均值和标准差用于后续反归一化
        self.inn_means = means
        self.inn_stds = stds
        
        # 为了保持兼容性，也存储min/max（使用全局统计）
        self.data_min = np.min(data)
        self.data_max = np.max(data)
        
        data_normalized = np.expand_dims(data_normalized.T, axis=-1)  # 转置并增加维度 (T, N, 1)
        
        x_offsets = np.sort(np.concatenate((np.arange(-self.seq_length + 1, 1, 1),)))
        y_offsets = np.sort(np.arange(1, self.pred_length + 1, 1))
        
        x, y = [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_timesteps - abs(max(y_offsets)))
        
        for t in range(min_t, max_t):
            x_t = data_normalized[t + x_offsets, ...]
            y_t = data_normalized[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        
        return np.stack(x, axis=0), np.stack(y, axis=0)
    
    def inverse_normalize(self, normalized_data, method='instance'):
        """
        反归一化方法 - 保持接口名称不变，支持两种方式
        :param normalized_data: 归一化后的数据
        :param method: 'instance' 或 'global'，选择反归一化方法
        :return: 反归一化后的数据
        """
        if method == 'instance' and self.inn_means is not None and self.inn_stds is not None:
            # 使用实例归一化反归一化
            return self._inverse_instance_normalization(normalized_data)
        elif self.data_min is not None and self.data_max is not None:
            # 使用全局归一化反归一化（保持向后兼容）
            return normalized_data * (self.data_max - self.data_min + 1e-8) + self.data_min
        else:
            raise ValueError("没有可用的归一化参数")
    
    def _inverse_instance_normalization(self, normalized_data):
        """
        内部方法：反实例归一化
        """
        means = self.inn_means
        stds = self.inn_stds
        
        # 确保形状匹配
        if normalized_data.shape[-2] != means.shape[0]:  # 节点数维度
            raise ValueError(f"数据形状不匹配: 归一化数据有 {normalized_data.shape[-2]} 个节点，但均值和标准差对应 {means.shape[0]} 个节点")
        
        # 反归一化: x = x_norm * std + mean
        # 调整均值和标准差的形状以匹配数据
        if len(normalized_data.shape) == 3:  # (T, N, features)
            denormalized_data = normalized_data * stds.T + means.T
        elif len(normalized_data.shape) == 4:  # (batch, T, N, features)
            denormalized_data = normalized_data * stds.T + means.T
        else:
            raise ValueError(f"不支持的维度: {normalized_data.shape}")
        
        return denormalized_data
    
    def train_val_test_split(self, x, y, train_ratio=0.6, val_ratio=0.2):
        """数据集划分 - 保持接口不变"""
        num_samples = x.shape[0]
        num_test = round(num_samples * 0.2)
        num_train = round(num_samples * train_ratio)
        num_val = num_samples - num_test - num_train
        
        x_train, y_train = x[:num_train], y[:num_train]
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        x_test, y_test = x[-num_test:], y[-num_test:]
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
    def get_data_range(self):
        """获取数据范围用于反归一化 - 保持接口不变"""
        return self.data_min, self.data_max
    
    def get_instance_norm_params(self):
        """获取实例归一化参数"""
        return self.inn_means, self.inn_stds
        
import torch
from torch.utils.data import Dataset
import numpy as np


class TrafficImageDataset(Dataset):
    def __init__(self, x_images, x_sequences, y_data, spatial_features=None, time_info=None):
        self.x_images = x_images  # 图像数据 (num_samples, num_nodes, H, W)
        self.x_sequences = x_sequences  # 原始序列 (num_samples, seq_length, num_nodes, feature_dim)
        self.y_data = y_data  # 目标数据
        self.spatial_features = spatial_features  # 空间特征
        self.time_info = time_info  # 时间信息
        
        print(f"Dataset shapes:")
        print(f"  x_images: {x_images.shape}")
        print(f"  x_sequences: {x_sequences.shape}")
        print(f"  y_data: {y_data.shape}")
        if spatial_features is not None:
            print(f"  spatial_features: {spatial_features.shape}")
        if time_info is not None:
            print(f"  time_info: {time_info.shape}")

    def __getitem__(self, idx):
        images = self.x_images[idx]  # (num_nodes, H, W)
        sequences = self.x_sequences[idx]  # (seq_length, num_nodes, feature_dim)
        targets = self.y_data[idx]  # (pred_length, num_nodes, feature_dim)
        
        # 处理目标数据
        if targets.ndim == 3 and targets.shape[2] == 1:
            targets = targets.squeeze(-1)  # (pred_length, num_nodes)
        targets = targets.T  # (num_nodes, pred_length)
        
        # 如果预测长度大于1，只取最后一个时间步
        if targets.shape[1] > 1:
            targets = targets[:, -1:]  # (num_nodes, 1)
        
        # 空间特征
        if self.spatial_features is not None:
            spatial_feat = self.spatial_features[idx]  # (num_nodes, 8)
        else:
            spatial_feat = np.zeros((images.shape[0], 8), dtype=np.float32)
        
        # 时间信息
        if self.time_info is not None:
            time_info = self.time_info[idx]  # (seq_length, 3)
        else:
            time_info = np.zeros((sequences.shape[0], 3), dtype=np.float32)
        
        # 添加通道维度到图像
        images = np.expand_dims(images, axis=1)  # (num_nodes, 1, H, W)
        
        return (
            torch.FloatTensor(images),        # (num_nodes, 1, H, W)
            torch.FloatTensor(sequences),     # (seq_length, num_nodes, feature_dim)
            torch.FloatTensor(targets),       # (num_nodes, pred_length)
            torch.FloatTensor(spatial_feat),  # (num_nodes, 8)
            torch.FloatTensor(time_info)      # (seq_length, 3)
        )
    
    def __len__(self):
        return self.x_images.shape[0]


 
