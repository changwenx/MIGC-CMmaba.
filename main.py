import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
from spatio import SpatialFeatureProcessor, SpatialTemporalFeatureExactor
from data_preprocessing import TrafficDataProcessor,TrafficImageDataset
from multiscale_processor import MultiscaleTimeSeriesProcessor
from mamba_vision_model import TrafficMambaVision 
from trainer import TrafficTrainer
from st_fusion import EnhancedMultiScaleModel
import gc

def create_multiscale_images(x_data, img_size=64, batch_size=1000):

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
    
def process_single_dataset(config):
    """处理单个数据集的函数"""
    args = config['args']
    data_path = config['data_path']
    dataset_name = config['dataset_name']
    
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"Data path: {data_path}")
    print(f"{'='*60}")
    
    # 设置输出目录
    args.output_dir = f"results/{dataset_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Step 1: Data preprocessing...")
    
    # 初始化数据处理器
    data_processor = TrafficDataProcessor(seq_length=12, pred_length=1)

    print(f"Loading data from {data_path}...")
    raw_data = data_processor.load_npz_data(data_path)
    print(f"Raw data shape: {raw_data.shape}")

    # 生成序列数据
    x_data, y_data = data_processor.generate_seq2seq_data(raw_data)
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
    raw_timesteps = raw_data.shape[1]  # 原始数据的时间步数
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
    
    # 自动生成邻接矩阵路径
    data_dir = os.path.dirname(data_path)
    dataset_base_name = os.path.basename(data_path).replace('.npz', '')
    adjacency_path = os.path.join(data_dir, f"{dataset_base_name}_adjacency_matrix.csv")
    
    # 如果邻接矩阵文件不存在，使用默认路径
    if not os.path.exists(adjacency_path):
        adjacency_path = os.path.join("/project/data/", 
                                    dataset_name, f"{dataset_name}_adjacency_matrix.csv")
    print(f"Using adjacency matrix: {adjacency_path}")
    
    # 初始化空间特征处理器
    spatial_processor = SpatialFeatureProcessor(
        adjacency_path, 
        num_clusters=min(args.coarse_clusters, num_nodes)
    )
    
    # 执行聚类
    mapping_matrix, cluster_labels, node_ids = spatial_processor.perform_clustering()
    
    # 确保聚类结果与数据节点数匹配
    if mapping_matrix.shape[0] != num_nodes:
        print(f"Warning: Clustering result has {mapping_matrix.shape[0]} nodes, but data has {num_nodes} nodes")
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
        feature_dim = st_extractor.extract_features(x_train[:1].transpose(0, 2, 1)).shape[-1]
    except:
        feature_dim = args.spatial_dim
    print(f"Feature dimension: {feature_dim}")

    # 批量提取所有数据集的特征
    def extract_features(data):
        features = []
        for i in range(0, len(data), args.batch_size):
            batch = data[i:min(i+args.batch_size, len(data))]
            try:
                features.append(st_extractor.extract_features(batch))
            except:
                features.append(np.zeros((len(batch), num_nodes, feature_dim)))
        return np.concatenate(features)

    # 处理所有数据集
    datasets = [x_train, x_val, x_test]
    spatial_features = np.concatenate([
        extract_features(d.squeeze(-1).transpose(0, 2, 1)) for d in datasets
    ])
    print(f"Spatial features shape: {spatial_features.shape}")

    # 确保空间特征与数据集大小匹配
    total_samples = len(x_train) + len(x_val) + len(x_test)
    if len(spatial_features) < total_samples:
        # 填充不足的部分
        padding_length = total_samples - len(spatial_features)
        padding = np.zeros((padding_length, spatial_features.shape[1], spatial_features.shape[2]))
        spatial_features = np.concatenate([spatial_features, padding], axis=0)
    
    print("Step 4: Creating datasets...")
    
    # 创建数据集
    train_dataset = TrafficImageDataset(
        x_train_images, 
        x_train,
        y_train, 
        spatial_features[:len(y_train)], 
        x_time_info_train
    )
    
    val_dataset = TrafficImageDataset(
        x_val_images, 
        x_val,
        y_val, 
        spatial_features[len(y_train):len(y_train)+len(y_val)],
        x_time_info_val
    )
    
    test_dataset = TrafficImageDataset(
        x_test_images, 
        x_test,
        y_test, 
        spatial_features[len(y_train)+len(y_val):],
        x_time_info_test
    ) 

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("Step 5: Initializing model...")
    
    spatial_dim = spatial_features.shape[-1]
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
        'output_dir': args.output_dir
    }

    print("Step 6: Starting training...")

    data_min_val, data_max_val = data_processor.get_data_range()
    inn_means, inn_stds = data_processor.get_instance_norm_params()
    trainer = TrafficTrainer(model, train_loader, val_loader, test_loader, 
                       train_config, data_min_val, data_max_val, inn_means, inn_stds)

    # 修改这里：获取详细的测试结果
    test_results = trainer.train()  # 假设train()方法返回一个字典

    print(f"Training completed!")

    # 打印详细的测试指标
    if isinstance(test_results, dict):
        print("Detailed test results:")
        for metric, value in test_results.items():
            print(f"  {metric}: {value:.4f}")
        test_loss = test_results.get('test_loss', test_results.get('loss', 0.0))
    else:
        test_loss = test_results
        print(f"Final test loss: {test_loss:.4f}")

    return {
        'dataset_name': dataset_name,
        'test_loss': test_loss,
        'test_results': test_results if isinstance(test_results, dict) else None,
        'num_nodes': num_nodes,
        'model_parameters': sum(p.numel() for p in model.parameters())
    }

def main():
    """主函数：批量处理多个数据集"""
    
    # 定义要处理的数据集列表
    datasets_config = [
        {
            'data_path': "/data/PEMS03/PEMS03.npz",
            'dataset_name': "PEMS03",
            'coarse_clusters': 31,
            'epochs': 35
        },
       
    ]
    
    # 解析命令行参数（用于通用设置）
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--spatial_dim', type=int, default=32)
    parser.add_argument('--chebyshev_order', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=64)  
    parser.add_argument('--epochs', type=int, default=35) 
   
    
    base_args = parser.parse_args()
    
  
    results = {}
    failed_datasets = []
    
    print("Starting multi-dataset processing...")
    print(f"Total datasets to process: {len(datasets_config)}")
    
    for i, dataset_info in enumerate(datasets_config):
        dataset_name = dataset_info['dataset_name']
        
        
            
        # 检查数据文件是否存在
        if not os.path.exists(dataset_info['data_path']):
            print(f"\nDataset file not found: {dataset_info['data_path']}")
            failed_datasets.append(dataset_name)
            results[dataset_name] = None
            continue
        
        try:
            # 为每个数据集创建独立的参数配置
            dataset_args = argparse.Namespace(**vars(base_args))
            dataset_args.coarse_clusters = dataset_info['coarse_clusters']
            dataset_args.epochs = dataset_info['epochs']
            
            config = {
                'args': dataset_args,
                'data_path': dataset_info['data_path'],
                'dataset_name': dataset_name
            }
            
            # 处理单个数据集
            result = process_single_dataset(config)
            results[dataset_name] = result
            
            print(f"\n✓ Successfully processed {dataset_name}")
            
        except Exception as e:
            print(f"\n✗ Error processing {dataset_name}: {str(e)}")
            failed_datasets.append(dataset_name)
            results[dataset_name] = None
            
        # 添加间隔，使输出更清晰
        if i < len(datasets_config) - 1:
            print("\n" + "="*80 + "\n")
    
    # 打印汇总结果
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL DATASET RESULTS")
    print(f"{'='*60}")
    
    successful_count = 0
    for dataset_name, result in results.items():
        if result is not None:
            print(f"✓ {dataset_name}:")
            print(f"  Test Loss: {result['test_loss']:.4f}")
            print(f"  Number of Nodes: {result['num_nodes']}")
            print(f"  Model Parameters: {result['model_parameters']:,}")
            successful_count += 1
        else:
            print(f"✗ {dataset_name}: FAILED")
    
    print(f"\nSuccessful: {successful_count}/{len(datasets_config)}")
    if failed_datasets:
        print(f"Failed datasets: {failed_datasets}")
    
    # 保存总体结果到文件
    summary_path = "results/experiment_summary.txt"
    os.makedirs("results", exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write("Multi-Dataset Experiment Summary\n")
        f.write("="*50 + "\n")
        for dataset_name, result in results.items():
            if result is not None:
                f.write(f"{dataset_name}:\n")
                f.write(f"  Test Loss: {result['test_loss']:.4f}\n")
                f.write(f"  Number of Nodes: {result['num_nodes']}\n")
                f.write(f"  Model Parameters: {result['model_parameters']:,}\n")
            
                # 如果有详细的测试结果，也保存
                if result.get('test_results'):
                    f.write("  Detailed Metrics:\n")
                    for metric, value in result['test_results'].items():
                        f.write(f"    {metric}: {value:.4f}\n")
                f.write("\n")
            else:
                f.write(f"{dataset_name}: FAILED\n\n")



if __name__ == "__main__":
    main()
