import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

class TrafficTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, config, data_min_val, data_max_val, inn_means=None, inn_stds=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        self.data_min_val = torch.tensor(data_min_val, dtype=torch.float32)
        self.data_max_val = torch.tensor(data_max_val, dtype=torch.float32)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.data_min_val = self.data_min_val.to(self.device)
        self.data_max_val = self.data_max_val.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config['step_size'], 
            gamma=config['gamma']
        )
        
        # 损失函数配置
        self.loss_type = config.get('loss_type', 'mse')
        
        if self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif self.loss_type == 'huber':
            self.criterion = nn.SmoothL1Loss()
        elif self.loss_type == 'combined':
            self.mse_loss = nn.MSELoss()
            self.mae_loss = nn.L1Loss()
        elif self.loss_type == 'logcosh':
            self.criterion = self.logcosh_loss
        elif self.loss_type == 'weighted_mse':
            self.criterion = self.weighted_mse_loss
        
        self.best_val_loss = float('inf')
        self.output_dir = config['output_dir']
        self.inn_means = inn_means
        self.inn_stds = inn_stds
        os.makedirs(self.output_dir, exist_ok=True)

    def logcosh_loss(self, y_pred, y_true):
        """Log-cosh损失函数，对异常值比MSE更鲁棒"""
        return torch.mean(torch.log(torch.cosh(y_pred - y_true)))

    def weighted_mse_loss(self, y_pred, y_true):
        """加权MSE损失函数"""
        weights = torch.abs(y_true) + 1.0  # 避免零权重
        return torch.mean(weights * (y_pred - y_true) ** 2)

    def combined_loss(self, y_pred, y_true, alpha=0.7):
        """组合MSE和MAE损失"""
        mse_loss = self.mse_loss(y_pred, y_true)
        mae_loss = self.mae_loss(y_pred, y_true)
        return alpha * mse_loss + (1 - alpha) * mae_loss

    def mape_loss(self, y_pred, y_true, epsilon=1e-8):
        """MAPE损失函数"""
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    def smape_loss(self, y_pred, y_true, epsilon=1e-8):
        """sMAPE对称平均绝对百分比误差"""
        return torch.mean(2 * torch.abs(y_pred - y_true) / (torch.abs(y_pred) + torch.abs(y_true) + epsilon)) * 100

    def pinball_loss(self, y_pred, y_true, quantile=0.5):
        """分位数损失，用于不确定性估计"""
        error = y_true - y_pred
        return torch.mean(torch.max(quantile * error, (quantile - 1) * error))

    def calculate_loss(self, y_pred, y_true):
        """根据配置计算损失"""
        if self.loss_type == 'combined':
            return self.combined_loss(y_pred, y_true)
        elif self.loss_type == 'mape':
            return self.mape_loss(y_pred, y_true)
        elif self.loss_type == 'smape':
            return self.smape_loss(y_pred, y_true)
        elif self.loss_type == 'pinball':
            return self.pinball_loss(y_pred, y_true)
        else:
            return self.criterion(y_pred, y_true)

    def calculate_metrics(self, y_true, y_pred):
        """确保反归一化正确"""
        # 转换到numpy
        y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
        y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred

        print(f"Input shapes - y_true: {y_true_np.shape}, y_pred: {y_pred_np.shape}")
        print(f"归一化前范围 - y_true: {y_true_np.min():.6f} to {y_true_np.max():.6f}")
        print(f"归一化前范围 - y_pred: {y_pred_np.min():.6f} to {y_pred_np.max():.6f}")

        # ============ 正确的反归一化方法 ============
        # 方法1: 如果实例归一化参数可用，使用实例归一化
        if hasattr(self, 'inn_means') and hasattr(self, 'inn_stds'):
            print("使用实例归一化反归一化")
            inn_means = self.inn_means.cpu().numpy() if isinstance(self.inn_means, torch.Tensor) else self.inn_means
            inn_stds = self.inn_stds.cpu().numpy() if isinstance(self.inn_stds, torch.Tensor) else self.inn_stds
        
            # 确保形状匹配: (170, 1) -> (1, 170)
            inn_means = inn_means.reshape(1, -1)
            inn_stds = inn_stds.reshape(1, -1)
        
            y_true_denorm = y_true_np * inn_stds + inn_means
            y_pred_denorm = y_pred_np * inn_stds + inn_means
        
        # 方法2: 使用正确的全局归一化参数
        else:
            print("使用全局归一化反归一化")
            # 使用训练时保存的正确范围
            data_min = self.data_min_val.cpu().numpy() if isinstance(self.data_min_val, torch.Tensor) else self.data_min_val
            data_max = self.data_max_val.cpu().numpy() if isinstance(self.data_max_val, torch.Tensor) else self.data_max_val
            print(f"使用预设范围: min={data_min:.2f}, max={data_max:.2f}")
        
            data_range = data_max - data_min + 1e-8
            y_true_denorm = y_true_np * data_range + data_min
            y_pred_denorm = y_pred_np * data_range + data_min

        # 确保非负
        y_true_denorm = np.clip(y_true_denorm, 0, None)
        y_pred_denorm = np.clip(y_pred_denorm, 0, None)

        print(f"反归一化后范围 - True: {y_true_denorm.min():.2f} to {y_true_denorm.max():.2f}")
        print(f"反归一化后范围 - Pred: {y_pred_denorm.min():.2f} to {y_pred_denorm.max():.2f}")

        # 计算MAE和RMSE
        mae = np.mean(np.abs(y_true_denorm - y_pred_denorm))
        rmse = np.sqrt(np.mean((y_true_denorm - y_pred_denorm) ** 2))

        # ============ MAPE计算 ============
        mask = y_true_denorm > 0
        zero_count = np.sum(~mask)
        non_zero_count = np.sum(mask)
    
        if non_zero_count > 0:
            y_true_nonzero = y_true_denorm[mask]
            y_pred_nonzero = y_pred_denorm[mask]
        
            relative_errors = np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)
        
            # 过滤异常值
            median_error = np.median(relative_errors)
            relative_errors = np.where(relative_errors > 40.0, median_error, relative_errors)
        
            mape = np.mean(relative_errors) * 100
        else:
            mape = 0.0

        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        print(f"零流量样本: {zero_count}, MAE: {np.mean(np.abs(y_pred_denorm[~mask])):.2f}")

        return mae, rmse, mape
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # 接收5个值：x_images, x_sequences, y, spatial_features, time_info
            x_images, x_sequences, y, spatial_features, time_info = batch
    
            x_images = x_images.to(self.device)
            x_sequences = x_sequences.to(self.device)
            y = y.to(self.device)
            spatial_features = spatial_features.to(self.device)
            time_info = time_info.to(self.device)
    
            # 传入时间信息
            output, _ = self.model(x_images, x_sequences, spatial_features, time_info)
    
            # 确保目标数据格式正确
            if y.dim() == 3 and y.shape[2] == 1:
                y = y.squeeze(-1)  # (B, N)
    
            self.optimizer.zero_grad()
    
            # 确保输出和目标形状匹配
            if output.shape != y.shape:
                output = output.squeeze(-1) if output.dim() > y.dim() else output
    
            # 使用新的损失函数计算方法
            loss = self.calculate_loss(output, y)
            loss.backward()
        
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()
        
            total_loss += loss.item()
            all_preds.append(output.detach())
            all_targets.append(y.detach())
    
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        train_mae, train_rmse, train_mape = self.calculate_metrics(all_targets, all_preds)
    
        return total_loss / len(self.train_loader), train_mae, train_rmse, train_mape

    def validate(self, loader):
        """验证函数"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
    
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation"):
                x_images, x_sequences, y, spatial_features, time_info = batch
            
                x_images = x_images.to(self.device)
                x_sequences = x_sequences.to(self.device)
                y = y.to(self.device)
                spatial_features = spatial_features.to(self.device)
                time_info = time_info.to(self.device)
            
                # 传入时间信息
                output, _ = self.model(x_images, x_sequences, spatial_features, time_info)
            
                if y.dim() == 3 and y.shape[2] == 1:
                    y = y.squeeze(-1)
            
                if output.shape != y.shape:
                    output = output.squeeze(-1) if output.dim() > y.dim() else output
            
                # 使用新的损失函数计算方法
                loss = self.calculate_loss(output, y)
                total_loss += loss.item()
            
                all_preds.append(output.detach())
                all_targets.append(y.detach())
    
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        mae, rmse, mape = self.calculate_metrics(all_targets, all_preds)
    
        return total_loss / len(loader), mae, rmse, mape

    def train(self):
        """训练主函数"""
        print(f"Using loss function: {self.loss_type}")
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # 训练
            train_loss, train_mae, train_rmse, train_mape = self.train_epoch()
            
            # 验证
            val_loss, val_mae, val_rmse, val_mape = self.validate(self.val_loader)
            
            print(f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.2f}%")
            
            # 学习率调度
            self.scheduler.step()
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))
                print("Saved best model!")
            
            # 定期保存检查点
            if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"Saved checkpoint at epoch {epoch+1}")
        
        # 加载最佳模型进行测试
        print("Loading best model for testing...")
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, 'best_model.pth')))
        
        # 测试
        test_loss, test_mae, test_rmse, test_mape = self.validate(self.test_loader)
        print(f"Test Results - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.2f}%")
        
        # 保存最终测试结果
        results = {
            'test_loss': test_loss,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'loss_type': self.loss_type
        }
        
        results_path = os.path.join(self.output_dir, 'test_results.npy')
        np.save(results_path, results)
        print(f"Saved test results to {results_path}")
        
        return test_loss
