import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from spatio import SpatialFeatureProcessor, SpatialTemporalFeatureExactor
from data_preprocessing import TrafficDataProcessor, TrafficImageDataset
from multiscale_processor import MultiscaleTimeSeriesProcessor
from mamba_vision_model import TrafficMambaVision 
from trainer import TrafficTrainer
from st_fusion import EnhancedMultiScaleModel
import gc


def create_multiscale_images(x_data, img_size=128, batch_size=1000):
    image_generator = MultiscaleTimeSeriesProcessor(
        scales=[1, 2, 6, 12],
        img_height=img_size,
        img_width=img_size
    )
    
    num_samples = len(x_data)
    num_nodes = x_data.shape[2]
    
    all_images = []
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_data = x_data[start_idx:end_idx]
        
        print(f"Processing batch {start_idx}-{end_idx}...")
        batch_images = image_generator.process_batch(batch_data)
        all_images.append(batch_images)
    
    return np.concatenate(all_images, axis=0)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing dataset.npy and matrix.npy')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--spatial_dim', type=int, default=32, help='Spatial feature dimension')
    parser.add_argument('--chebyshev_order', type=int, default=3, help='Order of Chebyshev polynomials')
    parser.add_argument('--coarse_clusters', type=int, default=31, help='Number of coarse clusters')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Step 1: Data preprocessing...")
    
    # 初始化数据处理器
    data_processor = TrafficDataProcessor(seq_length=12, pred_length=1)

    print(f"Loading PEMS-Bay data from {args.dataset_dir}...")
    
    # 加载PEMS-Bay数据
    dataset_path = os.path.join(args.dataset_dir, "dataset.npy")
    matrix_path = os.path.join(args.dataset_dir, "matrix.npy")
    
    # 加载数据
    raw_data = np.load(dataset_path)  # 形状应该是 (timesteps, num_nodes, num_features)
    adjacency_matrix = np.load(matrix_path)  # 邻接矩阵
    
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
    
    # PEMS-Bay数据形状为 (52116, 325, 2) - 这已经是正确的形状了
    # 不需要转置，直接使用
    # 只使用第一个特征（流量数据），忽略第二个特征（如果有的话）
    if raw_data.shape[-1] > 1:
        print(f"Using only the first feature from {raw_data.shape[-1]} features")
        raw_data = raw_data[:, :, 0:1]  # 只取第一个特征，保持3维
    
    print(f"Processed raw data shape: {raw_data.shape}")

    # 生成序列数据 - 现在数据是3维的，需要检查data_preprocessing能否处理
    try:
        x_data, y_data = data_processor.generate_seq2seq_data(raw_data)
    except ValueError as e:
        print(f"Error with 3D data: {e}")
        print("Trying to reshape data to 2D...")
        # 如果data_preprocessing只能处理2D数据，进行reshape
        # 将 (timesteps, nodes, features) 转为 (nodes, timesteps)
        if raw_data.shape[-1] == 1:
            raw_data_2d = raw_data.squeeze(-1).T  # 转为 (nodes, timesteps)
            print(f"Reshaped data to 2D: {raw_data_2d.shape}")
            x_data, y_data = data_processor.generate_seq2seq_data(raw_data_2d)
        else:
            raise e
    
    print(f"x_data shape: {x_data.shape}, y_data shape: {y_data.shape}")
    
    # 数据集划分
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = \
        data_processor.train_val_test_split(x_data, y_data)
    
    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    def create_time_info(x_data, raw_timesteps):
        """创建时间信息矩阵 - 修改为只包含时间和日期信息"""
        num_samples, seq_length, num_nodes, _ = x_data.shape
        time_info = np.zeros((num_samples, seq_length, 2), dtype=np.float32)  # 改为2维
    
        for i in range(num_samples):
            for t in range(seq_length):
                global_time_step = i * seq_length + t
                if global_time_step < raw_timesteps:
                    time_info[i, t, 0] = (global_time_step % 288) / 287.0  # 小时
                    time_info[i, t, 1] = ((global_time_step // 288) % 7) / 6.0  # 星期几
        return time_info

    # 创建时间信息
    raw_timesteps = raw_data.shape[0]  # 原始数据的时间步数
    x_time_info_train = create_time_info(x_train, raw_timesteps)
    x_time_info_val = create_time_info(x_val, raw_timesteps)
    x_time_info_test = create_time_info(x_test, raw_timesteps)
    
    print(f"Time info shapes - Train: {x_time_info_train.shape}, Val: {x_time_info_val.shape}, Test: {x_time_info_test.shape}")

    print("Step 2: Creating multiscale time series images...")
    processor = MultiscaleTimeSeriesProcessor(img_height=64, img_width=64)

    # 处理图像（使用原始的时间序列数据）
    x_train_images = processor.process_batch(x_train)
    x_val_images = processor.process_batch(x_val)  
    x_test_images = processor.process_batch(x_test)

    print(f"Image shapes - Train: {x_train_images.shape}, Val: {x_val_images.shape}, Test: {x_test_images.shape}")

    print("Step 3: Processing spatial features...")
    
    # 检查节点数量
    num_nodes = x_train.shape[2]
    print(f"Number of nodes: {num_nodes}")
    
    # 保存邻接矩阵为CSV文件（供SpatialFeatureProcessor使用）
    adjacency_csv_path = os.path.join(args.output_dir, "pems_bay_adjacency.csv")
    pd.DataFrame(adjacency_matrix).to_csv(adjacency_csv_path, index=False, header=False)
    print(f"Saved adjacency matrix to {adjacency_csv_path}")
    
    # 初始化空间特征处理器
    spatial_processor = SpatialFeatureProcessor(
        adjacency_csv_path, 
        num_clusters=min(args.coarse_clusters, num_nodes)
    )
    
    # 执行聚类
    mapping_matrix, cluster_labels, node_ids = spatial_processor.perform_clustering()
    
    # 确保聚类结果与数据节点数匹配
    if mapping_matrix.shape[0] != num_nodes:
        print(f"Warning: Clustering result has {mapping_matrix.shape[0]} nodes, but data has {num_nodes} nodes")
        # 如果聚类不匹配，使用单位矩阵
        mapping_matrix = np.eye(num_nodes, dtype=np.float32)
        spatial_processor.mapping_matrix = mapping_matrix
    
    # 初始化时空特征提取器
    st_extractor = SpatialTemporalFeatureExactor(
        spatial_processor, 
        input_dim=12,
        hidden_dim=64,
        output_dim=args.spatial_dim,
        chebyshev_order=args.chebyshev_order
    )
    
    # 提取空间特征
    print("Extracting spatial-temporal features...")
    
    # 确定特征维度
    try:
        # 调整输入形状以匹配提取器的期望
        sample_input = x_train[:1].squeeze(-1).transpose(0, 2, 1)  # (batch, nodes, seq_len)
        feature_dim = st_extractor.extract_features(sample_input).shape[-1]
    except Exception as e:
        print(f"Error extracting sample features: {e}")
        feature_dim = args.spatial_dim
    
    print(f"Feature dimension: {feature_dim}")

    # 批量提取所有数据集的特征
    def extract_features(data):
        features = []
        batch_size = min(args.batch_size, len(data))  # 防止batch_size大于数据量
        for i in range(0, len(data), batch_size):
            batch = data[i:min(i+batch_size, len(data))]
            try:
                # 调整输入形状
                batch_reshaped = batch.squeeze(-1).transpose(0, 2, 1)  # (batch, nodes, seq_len)
                features.append(st_extractor.extract_features(batch_reshaped))
            except Exception as e:
                print(f"Error extracting features for batch {i}: {e}")
                # 创建零特征作为后备
                features.append(np.zeros((len(batch), num_nodes, feature_dim)))
        return np.concatenate(features)

    # 分别处理每个数据集以避免内存问题
    print("Extracting training features...")
    train_spatial_features = extract_features(x_train)
    print("Extracting validation features...")
    val_spatial_features = extract_features(x_val)
    print("Extracting test features...")
    test_spatial_features = extract_features(x_test)
    
    print(f"Spatial features shapes - Train: {train_spatial_features.shape}, Val: {val_spatial_features.shape}, Test: {test_spatial_features.shape}")

    print("Step 4: Creating datasets...")
    
    # 创建数据集
    train_dataset = TrafficImageDataset(
        x_train_images, 
        x_train,  # 原始时间序列数据 (num_samples, seq_length, num_nodes, 1)
        y_train, 
        train_spatial_features, 
        x_time_info_train
    )
    
    val_dataset = TrafficImageDataset(
        x_val_images, 
        x_val,
        y_val, 
        val_spatial_features,
        x_time_info_val
    )
    
    test_dataset = TrafficImageDataset(
        x_test_images, 
        x_test,
        y_test, 
        test_spatial_features,
        x_time_info_test
    ) 

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("Step 5: Initializing model...")
    
    spatial_dim = train_spatial_features.shape[-1]
    # 初始化增强模型
    model = EnhancedMultiScaleModel(
        spatial_dim=spatial_dim,
        in_chans=1,
        num_classes=1,
        depths=[3, 3, 2, 2],
        dims=[64, 128, 256, 512, 1024],
        window_sizes=[8, 8, 14, 7],
        mlp_ratio=4,
        drop_rate=0.1,
        drop_path_rate=0.2,
        layer_scale=1e-5,
        seq_length=12,  # 序列长度
        feature_dim=1  # 特征维度 
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 训练配置
    train_config = {
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'step_size': 20,
        'gamma': 0.5,
        'max_grad_norm': 1.0,
        'epochs': args.epochs,
        'checkpoint_interval': 10,
        'output_dir': args.output_dir
    }

    print("Step 6: Starting training...")
    
    data_min_val, data_max_val = data_processor.get_data_range()
    inn_means, inn_stds = data_processor.get_instance_norm_params()
    trainer = TrafficTrainer(model, train_loader, val_loader, test_loader, 
                           train_config, data_min_val, data_max_val, inn_means, inn_stds)
    test_loss = trainer.train()

    print(f"Training completed! Final test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
