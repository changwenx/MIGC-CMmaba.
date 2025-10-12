import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

class MultiscaleTimeSeriesProcessor:
    """多尺度时间序列处理器 - 热力图版本（保持输出形状）"""
    
    def __init__(self, scales=[1, 2, 6, 12], img_height=128, img_width=128):
        self.scales = scales
        self.img_height = img_height
        self.img_width = img_width
        self.num_scales = len(scales)
    
    def generate_multiscale_heatmap(self, time_series):
        """生成单张多尺度热力图 - 保持(128,128)输出形状"""
        # 创建单通道图像，4个尺度分别放在4个区域
        image = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        rows_per_scale = self.img_height // self.num_scales
        
        for scale_idx, scale in enumerate(self.scales):
            start_row = scale_idx * rows_per_scale
            end_row = start_row + rows_per_scale
            
            # 生成尺度数据
            scale_data = self._generate_scale_data(time_series, scale)
            
            if len(scale_data) > 0:
                # 绘制热力图到对应行区域
                self._draw_heatmap_to_rows(image, scale_data, start_row, end_row, scale)
        
        return image
    
    def _generate_scale_data(self, time_series, scale):
        """生成尺度数据 - 使用滑动窗口求和（保持不变）"""
        if scale == 1:
            return time_series
        
        seq_length = len(time_series)
        
        if seq_length % scale != 0:
            pad_width = scale - (seq_length % scale)
            padded_series = np.pad(time_series, (0, pad_width), 'constant')
        else:
            padded_series = time_series
        
        window_size = scale
        num_windows = len(padded_series) // window_size
        
        if num_windows == 0:
            return np.array([])
        
        indices = np.arange(0, len(padded_series), window_size)
        scale_data = np.add.reduceat(padded_series, indices)[:num_windows]
        
        return scale_data
    
    def _draw_heatmap_to_rows(self, image, scale_data, start_row, end_row, scale):
        if len(scale_data) == 0:
            return
    
        # 归一化
        norm_data = (scale_data - np.min(scale_data)) / (np.max(scale_data) - np.min(scale_data) + 1e-8)
        x_positions = np.linspace(0, self.img_width - 1, len(norm_data)).astype(int)

        # 生成衰减 mask (一次性生成行的衰减系数)
        rows = np.arange(start_row, end_row)[:, None]  # shape=(rows,1)
        center_y = start_row + (end_row - start_row) // 2
        decay = 1 - np.abs(rows - center_y) / ((end_row - start_row) / 2)
        decay = np.clip(decay, 0, 1)  # shape=(rows,1)

        # 每个点的强度
        intensities = (255 * norm_data).astype(np.uint8)  # shape=(len(x_positions),)
    
        # 批量写入 image
        for x, base_intensity in zip(x_positions, intensities):
            col_values = (base_intensity * decay).astype(np.uint8).squeeze()
            image[start_row:end_row, x] = np.maximum(image[start_row:end_row, x], col_values)

    
    def process_batch(self, x_data):
        """
        批量处理 (num_samples, seq_length, num_nodes, 1) -> (num_samples, num_nodes, H, W)
        保持完全相同的输入输出格式
        """
        num_samples, seq_length, num_nodes, _ = x_data.shape
        images = np.zeros((num_samples, num_nodes, self.img_height, self.img_width), dtype=np.uint8)
        
        # 重塑为 (num_samples * num_nodes, seq_length)
        flat_data = x_data.reshape(-1, seq_length)
        total_sequences = len(flat_data)
        
        print(f"Processing {total_sequences} sequences with heatmap approach...")
        
        # 批量处理
        batch_size = 2000
        for start_idx in range(0, total_sequences, batch_size):
            end_idx = min(start_idx + batch_size, total_sequences)
            batch_data = flat_data[start_idx:end_idx]
            
            # 批量生成热力图
            for j, seq in enumerate(batch_data):
                global_idx = start_idx + j
                sample_idx = global_idx // num_nodes
                node_idx = global_idx % num_nodes
                
                # 生成多尺度热力图
                img = self.generate_multiscale_heatmap(seq)
                images[sample_idx, node_idx] = img
            
            if end_idx % 10000 == 0 or end_idx == total_sequences:
                print(f"Processed {end_idx}/{total_sequences} sequences")
        
        return images

 
        

