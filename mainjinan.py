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
    
def create_time_info(x_data, raw_timesteps):
    """创建时间信息矩阵"""
    num_samples, seq_length, num_nodes, _ = x_data.shape
    time_info = np.zeros((num_samples, seq_length, 2), dtype=np.float32)
    
    for i in range(num_samples):
        for t in range(seq_length):
            global_time_step = i * seq_length + t
            if global_time_step < raw_timesteps:
                time_info[i, t, 0] = (global_time_step % 288) / 287.0  # 小时
                time_info[i, t, 1] = ((global_time_step // 288) % 7) / 6.0  # 星期几
    return time_info

def create_adjacency_from_coordinates(coords_path, num_nodes):
    """从坐标文件创建邻接矩阵"""
    print(f"Creating adjacency matrix from coordinates: {coords_path}")
    try:
        # 读取坐标数据
        df = pd.read_csv(coords_path)
        print(f"Coordinates data shape: {df.shape}")
        
        # 提取经纬度
        if 'longitude' in df.columns and 'latitude' in df.columns:
            coords = df[['longitude', 'latitude']].values
        else:
            # 尝试自动检测列
            coords = df.iloc[:, 1:3].values  # 假设第2,3列是经纬度
        
        coords = coords.astype(np.float32)
        print(f"Coordinates shape: {coords.shape}")
        
        # 如果坐标数量不匹配，使用前num_nodes个
        if coords.shape[0] > num_nodes:
            coords = coords[:num_nodes]
            print(f"Using first {num_nodes} coordinates")
        elif coords.shape[0] < num_nodes:
            # 填充不足的坐标
            padding = np.random.randn(num_nodes - coords.shape[0], 2) * 0.01 + coords.mean(axis=0)
            coords = np.vstack([coords, padding])
            print(f"Padded coordinates to {num_nodes}")
        
        # 计算距离矩阵
        from scipy.spatial.distance import cdist
        distance_matrix = cdist(coords, coords, metric='euclidean')
        
        # 创建邻接矩阵（基于距离的高斯核）
        sigma = np.percentile(distance_matrix, 15)
        threshold = np.percentile(distance_matrix, 25)
        
        adjacency_matrix = np.exp(-distance_matrix ** 2 / (2 * sigma ** 2))
        adjacency_matrix[distance_matrix > threshold] = 0
        np.fill_diagonal(adjacency_matrix, 0)
        
        # 行归一化
        row_sum = adjacency_matrix.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        adjacency_matrix = adjacency_matrix / row_sum
        
        print(f"Adjacency matrix created: {adjacency_matrix.shape}")
        return adjacency_matrix
        
    except Exception as e:
        print(f"Error creating adjacency matrix: {e}")
        print("Using identity matrix as fallback")
        return np.eye(num_nodes, dtype=np.float32)

def perform_clustering(adjacency_matrix, num_clusters):
    """执行聚类"""
    print(f"Performing clustering with {num_clusters} clusters...")
    
    try:
        from sklearn.cluster import SpectralClustering
        
        n_clusters = min(num_clusters, adjacency_matrix.shape[0] - 1)
        if n_clusters < 2:
            n_clusters = 2
            
        # 确保对称
        adj_sym = (adjacency_matrix + adjacency_matrix.T) / 2
        
        spectral_clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42,
            assign_labels='discretize'
        )
        
        cluster_labels = spectral_clustering.fit_predict(adj_sym)
        mapping_matrix = np.eye(n_clusters, dtype=np.float32)[cluster_labels]
        
        print(f"Clustering completed: {n_clusters} clusters")
        return mapping_matrix, cluster_labels
        
    except Exception as e:
        print(f"Clustering failed: {e}, using uniform clustering")
        n_nodes = adjacency_matrix.shape[0]
        cluster_labels = np.arange(n_nodes) % num_clusters
        mapping_matrix = np.eye(num_clusters, dtype=np.float32)[cluster_labels]
        return mapping_matrix, cluster_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="/data/Jinan/JiNan.npz")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--spatial_dim', type=int, default=32, help='Spatial feature dimension')
    parser.add_argument('--chebyshev_order', type=int, default=3, help='Order of Chebyshev polynomials')
    parser.add_argument('--coarse_clusters', type=int, default=31, help='Number of coarse clusters')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Step 1: Data preprocessing...")
    
    # 初始化数据处理器
    data_processor = TrafficDataProcessor(seq_length=12, pred_length=1)

    print(f"Loading data from {args.data_path}...")
    raw_data = data_processor.load_npz_data(args.data_path)
    print(f"Raw data shape: {raw_data.shape}")

    # 生成序列数据
    x_data, y_data = data_processor.generate_seq2seq_data(raw_data)
    print(f"x_data shape: {x_data.shape}, y_data shape: {y_data.shape}")
    
    # 数据集划分
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = \
        data_processor.train_val_test_split(x_data, y_data)
    
    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    # 创建时间信息
    raw_timesteps = raw_data.shape[1]
    x_time_info_train = create_time_info(x_train, raw_timesteps)
    x_time_info_val = create_time_info(x_val, raw_timesteps)
    x_time_info_test = create_time_info(x_test, raw_timesteps)
    
    print(f"Time info shapes - Train: {x_time_info_train.shape}, Val: {x_time_info_val.shape}, Test: {x_time_info_test.shape}")

    print("Step 2: Creating multiscale time series images...")
    processor = MultiscaleTimeSeriesProcessor(img_height=64, img_width=64)

    x_train_images = processor.process_batch(x_train)
    x_val_images = processor.process_batch(x_val)  
    x_test_images = processor.process_batch(x_test)

    print(f"Image shapes - Train: {x_train_images.shape}, Val: {x_val_images.shape}, Test: {x_test_images.shape}")

    print("Step 3: Processing spatial features...")
    
    # 检查节点数量
    num_nodes = x_train.shape[2]
    print(f"Number of nodes: {num_nodes}")
    
    # 坐标文件路径
    coords_path = os.path.join("/project/data/Jinan/", "JiNan of lalo.csv")
    
    # 步骤3.1: 创建邻接矩阵
    adjacency_matrix = create_adjacency_from_coordinates(coords_path, num_nodes)
    
    # 步骤3.2: 执行聚类
    num_clusters = min(args.coarse_clusters, num_nodes)
    mapping_matrix, cluster_labels = perform_clustering(adjacency_matrix, num_clusters)
    node_ids = np.arange(num_nodes)
    
    print(f"Mapping matrix shape: {mapping_matrix.shape}")
    
    # 步骤3.3: 初始化空间处理器（但绕过其文件读取）
    spatial_processor = SpatialFeatureProcessor(
        coords_path, 
        num_clusters=num_clusters
    )
    
    # 直接设置处理好的矩阵
    spatial_processor.adjacency_matrix = adjacency_matrix
    spatial_processor.mapping_matrix = mapping_matrix
    
    # 步骤3.4: 初始化时空特征提取器
    st_extractor = SpatialTemporalFeatureExactor(
        spatial_processor, 
        input_dim=12,
        hidden_dim=64,
        output_dim=args.spatial_dim,
        chebyshev_order=args.chebyshev_order
    )
    
    # 提取空间特征
    print("Extracting spatial-temporal features...")
    
    # 批量提取特征
    def extract_features_batch(data):
        features = []
        batch_size = min(args.batch_size, 100)  # 防止内存溢出
        for i in range(0, len(data), batch_size):
            batch = data[i:min(i+batch_size, len(data))]
            try:
                batch_features = st_extractor.extract_features(batch)
                features.append(batch_features)
            except Exception as e:
                print(f"Error extracting features for batch {i}: {e}")
                # 创建默认特征
                default_features = np.zeros((len(batch), num_nodes, args.spatial_dim))
                features.append(default_features)
        return np.concatenate(features, axis=0)
    
    # 处理训练集
    print("Extracting features for training set...")
    x_train_transposed = x_train.squeeze(-1).transpose(0, 2, 1)
    spatial_features_train = extract_features_batch(x_train_transposed)
    
    # 处理验证集
    print("Extracting features for validation set...")
    x_val_transposed = x_val.squeeze(-1).transpose(0, 2, 1)
    spatial_features_val = extract_features_batch(x_val_transposed)
    
    # 处理测试集
    print("Extracting features for test set...")
    x_test_transposed = x_test.squeeze(-1).transpose(0, 2, 1)
    spatial_features_test = extract_features_batch(x_test_transposed)
    
    print(f"Spatial features - Train: {spatial_features_train.shape}, "
          f"Val: {spatial_features_val.shape}, Test: {spatial_features_test.shape}")
    
    print("Step 4: Creating datasets...")
    
    # 创建数据集
    train_dataset = TrafficImageDataset(
        x_train_images, 
        x_train,
        y_train, 
        spatial_features_train, 
        x_time_info_train
    )
    
    val_dataset = TrafficImageDataset(
        x_val_images, 
        x_val,
        y_val, 
        spatial_features_val,
        x_time_info_val
    )
    
    test_dataset = TrafficImageDataset(
        x_test_images, 
        x_test,
        y_test, 
        spatial_features_test,
        x_time_info_test
    ) 

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("Step 5: Initializing model...")
    
    spatial_dim = spatial_features_train.shape[-1]
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
        seq_length=12,
        feature_dim=1
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
        'output_dir': args.output_dir,
        'loss_type': 'combined'
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
