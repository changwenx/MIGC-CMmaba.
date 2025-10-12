import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChebyshevGCN(nn.Module):
    """切比雪夫图卷积网络"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, chebyshev_order=3):
        super(ChebyshevGCN, self).__init__()
        self.chebyshev_order = chebyshev_order
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 切比雪夫多项式系数
        self.chebyshev_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(input_dim, hidden_dim))
            for _ in range(chebyshev_order + 1)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for weight in self.chebyshev_weights:
            nn.init.xavier_uniform_(weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
    
    def compute_chebyshev_polynomials(self, L, X, order):
        """计算切比雪夫多项式"""
        # L: 归一化的拉普拉斯矩阵 [N, N]
        # X: 输入特征 [B, N, F]
        
        T_k = []
        T_0 = X  # T_0(X) = X
        T_1 = torch.matmul(L, X)  # T_1(X) = L·X
        
        T_k.append(T_0)
        if order >= 1:
            T_k.append(T_1)
        
        for k in range(2, order + 1):
            T_k_minus_1 = T_k[-1]
            T_k_minus_2 = T_k[-2]
            T_k_current = 2 * torch.matmul(L, T_k_minus_1) - T_k_minus_2  # 递推关系
            T_k.append(T_k_current)
        
        return T_k
    
    def forward(self, x, L):
        """
        x: 输入特征 [B, N, F]
        L: 归一化的拉普拉斯矩阵 [N, N]
        """
        B, N, F = x.shape
        
        # 确保拉普拉斯矩阵在正确设备上
        L = L.to(x.device)
        
        # 计算切比雪夫多项式
        chebyshev_basis = self.compute_chebyshev_polynomials(L, x, self.chebyshev_order)
        
        # 应用切比雪夫卷积
        output = torch.zeros(B, N, self.hidden_dim, device=x.device)
        
        for k in range(self.chebyshev_order + 1):
            # 对每个切比雪夫基应用权重
            weighted = torch.matmul(chebyshev_basis[k], self.chebyshev_weights[k])
            output += weighted
        
        output = self.activation(output)
        output = self.dropout(output)
        output = self.output_layer(output)
        
        return output
        
class SpatialFeatureProcessor:
    def __init__(self, adjacency_csv_path, num_clusters=31):
        self.adjacency_csv_path = adjacency_csv_path
        self.target_clusters = num_clusters
        self.mapping_matrix = None
        self.node_ids = None
        self.cluster_labels = None
        self.adjacency_matrix = None
        self.laplacian = None  # 新增：存储拉普拉斯矩阵

    def load_adjacency_data(self):
        try:
            df = pd.read_csv(self.adjacency_csv_path, header=None)
            df.columns = df.columns.astype(str).str.strip()
            print(f"Loaded adjacency CSV: {df.shape}, columns={df.columns.tolist()}")

            if set(['from', 'to']).issubset(df.columns):
                print("Detected edge list format.")
                weight_col = 'distance' if 'distance' in df.columns else 'cost'
                if weight_col not in df.columns:
                    print("No distance/cost column found, using uniform weights")
                    df[weight_col] = 1.0
                
                from_nodes = df['from'].unique()
                to_nodes = df['to'].unique()
                self.node_ids = np.unique(np.concatenate([from_nodes, to_nodes]))
                num_nodes = len(self.node_ids)
                print(f"Unique nodes: {num_nodes}")
                
                node_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
                adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
                
                for _, row in df.iterrows():
                    try:
                        i = node_to_idx[row['from']]
                        j = node_to_idx[row['to']]
                        weight = float(row[weight_col])
                        adjacency_matrix[i, j] = weight
                        adjacency_matrix[j, i] = weight
                    except KeyError as e:
                        continue
                
                self.adjacency_matrix = adjacency_matrix
                self._compute_laplacian()
                return adjacency_matrix, self.node_ids

            else:
                print("Detected adjacency matrix format.")
                adjacency_matrix = df.values.astype(np.float32)
                num_nodes = adjacency_matrix.shape[0]
                self.node_ids = np.arange(num_nodes)
                self.adjacency_matrix = adjacency_matrix
                self._compute_laplacian()
                return adjacency_matrix, self.node_ids
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise

    def _compute_laplacian(self):
        """计算归一化拉普拉斯矩阵"""
        if self.adjacency_matrix is None:
            raise ValueError("Adjacency matrix not loaded")
        
        A = self.adjacency_matrix
        # 计算度矩阵
        D = np.diag(np.sum(A, axis=1))
        
        # 避免除零错误，使用伪逆
        D_sqrt = np.sqrt(D)
        D_sqrt_inv = np.linalg.pinv(D_sqrt)
        
        # 计算归一化拉普拉斯矩阵: L = I - D^(-1/2) A D^(-1/2)
        L = np.eye(A.shape[0]) - D_sqrt_inv @ A @ D_sqrt_inv
        
        self.laplacian = L
        print(f"Computed normalized Laplacian matrix: {L.shape}")

    def get_laplacian(self):
        """获取拉普拉斯矩阵"""
        if self.laplacian is None:
            self._compute_laplacian()
        return self.laplacian

    def get_reduced_laplacian(self):
        """获取粗粒度级别的拉普拉斯矩阵"""
        if self.mapping_matrix is None:
            raise ValueError("Please perform clustering first!")
        
        # 聚合邻接矩阵到粗粒度级别
        A_coarse = self.mapping_matrix.T @ self.adjacency_matrix @ self.mapping_matrix
        
        # 计算粗粒度的拉普拉斯矩阵
        D_coarse = np.diag(np.sum(A_coarse, axis=1))
        D_coarse_sqrt = np.sqrt(D_coarse)
        D_coarse_sqrt_inv = np.linalg.pinv(D_coarse_sqrt)
        
        L_coarse = np.eye(A_coarse.shape[0]) - D_coarse_sqrt_inv @ A_coarse @ D_coarse_sqrt_inv
        
        return L_coarse

    def granular_ball_clustering(self, adjacency_matrix):

        num_nodes = adjacency_matrix.shape[0]
        
        # 计算相似度矩阵
        
        sigma = np.percentile(adjacency_matrix[adjacency_matrix > 0], 50)  # 使用中位数
        print(f"Sigma (median of non-zero distances): {sigma:.6f}")
    
        # 如果sigma太小，使用一个最小值
        if sigma < 1e-6:
            sigma = 1.0  # 避免除零
    
        similarity_matrix = np.exp(-adjacency_matrix**2 / (2 * sigma**2))
        
        # 初始化
        visited = np.zeros(num_nodes, dtype=bool)
        cluster_labels = -np.ones(num_nodes, dtype=int)
        current_cluster_id = 0
        
        # 粒球聚类的核心指标
        density_threshold = 0.7  # 提高密度阈值
        purity_threshold = 0.8   # 提高纯度阈值
        
        print(f"Using granular ball clustering with density_threshold={density_threshold}, purity_threshold={purity_threshold}")
        
        # 首先找到一些好的初始中心点（相似度高的区域）
        node_degrees = np.sum(adjacency_matrix > 0, axis=1)
        potential_centers = np.argsort(-node_degrees)[:min(50, num_nodes//2)]
        
        # 从潜在中心点开始聚类
        for center in potential_centers:
            if not visited[center]:
                # 创建初始粒球
                ball_members = [center]
                ball_density = 1.0
                ball_purity = 1.0
                visited[center] = True
                
                # 扩展粒球
                can_expand = True
                expansion_count = 0
                max_expansions = 10  # 限制扩展次数
                
                while can_expand and expansion_count < max_expansions:
                    can_expand = False
                    best_candidate = None
                    best_density = ball_density
                    best_purity = ball_purity
                    
                    # 寻找可以加入的候选节点
                    for candidate in range(num_nodes):
                        if not visited[candidate] and candidate not in ball_members:
                            # 计算候选节点与当前粒球的平均相似度
                            avg_similarity = np.mean([similarity_matrix[candidate, m] for m in ball_members])
                            
                            if avg_similarity > 0.6:  # 初步筛选
                                # 临时加入候选节点
                                temp_members = ball_members + [candidate]
                                
                                # 计算新密度
                                new_density = np.mean([
                                    similarity_matrix[i, j] 
                                    for i in temp_members 
                                    for j in temp_members 
                                    if i != j
                                ])
                                
                                # 计算新纯度
                                new_purity = np.min([
                                    similarity_matrix[i, j] 
                                    for i in temp_members 
                                    for j in temp_members 
                                    if i != j
                                ])
                                
                                # 检查是否满足粒球条件
                                if (new_density >= density_threshold and 
                                    new_purity >= purity_threshold and
                                    new_density >= best_density):
                                    
                                    best_candidate = candidate
                                    best_density = new_density
                                    best_purity = new_purity
                                    can_expand = True
                    
                    # 添加最佳候选节点
                    if best_candidate is not None:
                        ball_members.append(best_candidate)
                        visited[best_candidate] = True
                        ball_density = best_density
                        ball_purity = best_purity
                        expansion_count += 1
                
                # 分配聚类标签
                for member in ball_members:
                    cluster_labels[member] = current_cluster_id
                
                current_cluster_id += 1
                print(f"Cluster {current_cluster_id-1}: {len(ball_members)} nodes, density={ball_density:.3f}, purity={ball_purity:.3f}")
        
        # 处理未访问的节点（分配到最近的簇）
        unvisited_nodes = np.where(~visited)[0]
        for node in unvisited_nodes:
            # 找到最相似的已聚类节点
            max_similarity = -1
            nearest_cluster = -1
            
            for cluster_id in range(current_cluster_id):
                cluster_members = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_members) > 0:
                    avg_similarity = np.mean([similarity_matrix[node, m] for m in cluster_members])
                    if avg_similarity > max_similarity:
                        max_similarity = avg_similarity
                        nearest_cluster = cluster_id
            
            if nearest_cluster != -1:
                cluster_labels[node] = nearest_cluster
                visited[node] = True
        
        # 确保聚类数量接近目标
        unique_clusters = np.unique(cluster_labels)
        actual_clusters = len(unique_clusters)
        
        print(f"Initial clusters: {actual_clusters}, Target: {self.target_clusters}")
        
        # 如果聚类数量不足，将大簇拆分成小簇
        if actual_clusters < self.target_clusters:
            needed_clusters = self.target_clusters - actual_clusters
            clusters_to_split = []
            
            # 找到足够大的簇来拆分
            for cluster_id in unique_clusters:
                members = np.where(cluster_labels == cluster_id)[0]
                if len(members) >= 3:  # 至少3个节点才能拆分
                    clusters_to_split.append(cluster_id)
                    if len(clusters_to_split) >= needed_clusters:
                        break
            
            # 拆分选中的簇
            for i, cluster_id in enumerate(clusters_to_split):
                if current_cluster_id >= self.target_clusters:
                    break
                    
                members = np.where(cluster_labels == cluster_id)[0]
                if len(members) >= 3:
                    # 随机拆分（实际应用中可以用更智能的方法）
                    np.random.shuffle(members)
                    split_point = len(members) // 2
                    
                    # 将后半部分分配到新簇
                    for j in range(split_point, len(members)):
                        cluster_labels[members[j]] = current_cluster_id
                    
                    current_cluster_id += 1
        
        # 重新编号确保连续性
        unique_labels = np.unique(cluster_labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        cluster_labels = np.array([label_mapping[label] for label in cluster_labels])
        
        actual_clusters = len(unique_labels)
        print(f"Final clusters: {actual_clusters}")
        
        # 计算质量指标
        self.calculate_cluster_metrics(cluster_labels, similarity_matrix)
        
        return cluster_labels

    def calculate_cluster_metrics(self, cluster_labels, similarity_matrix):

        unique_clusters = np.unique(cluster_labels)
    
        densities = []
        purities = []
        sizes = []
    
        for cluster_id in unique_clusters:
            members = np.where(cluster_labels == cluster_id)[0]
            size = len(members)
            sizes.append(size)
        
            if size > 1:
                # 计算簇密度（平均相似度）
                cluster_similarities = []
                for i in range(size):
                    for j in range(i + 1, size):
                        cluster_similarities.append(similarity_matrix[members[i], members[j]])
            
                density = np.mean(cluster_similarities) if cluster_similarities else 0
                purity = np.min(cluster_similarities) if cluster_similarities else 0
            
                densities.append(density)
                purities.append(purity)
            
                #print(f"Cluster {cluster_id}: {size} nodes, density={density:.3f}, purity={purity:.3f}")
            else:
                print(f"Cluster {cluster_id}: {size} nodes (singleton)")
    
        if densities and purities:
            avg_density = np.mean(densities)
            avg_purity = np.mean(purities)
        
        else:
            print("one cluster")

    def create_mapping_matrix(self, cluster_labels):
    
        num_nodes = len(cluster_labels)
        unique_clusters = np.unique(cluster_labels)
        actual_clusters = len(unique_clusters)
        
        # 确保目标聚类数量与实际一致
        if actual_clusters != self.target_clusters:
            print(f"Adjusting target clusters from {self.target_clusters} to {actual_clusters}")
            self.target_clusters = actual_clusters
        
        print(f"\nCreating mapping matrix: {num_nodes} nodes -> {actual_clusters} clusters")
        
        mapping_matrix = np.zeros((num_nodes, actual_clusters), dtype=np.float32)
        for node_idx, cluster_id in enumerate(cluster_labels):
            mapping_matrix[node_idx, cluster_id] = 1.0
        
        # 检查映射矩阵是否正确
        print(f"Mapping matrix shape: {mapping_matrix.shape}")
        print(f"Mapping matrix sum per node: {mapping_matrix.sum(axis=1)}")  # 应该都是1.0
        print(f"Mapping matrix sum per cluster: {mapping_matrix.sum(axis=0)}")  # 每个簇的节点数
        
        self.mapping_matrix = mapping_matrix
        self.cluster_labels = cluster_labels
        return mapping_matrix

    def perform_clustering(self):
 
        #print("Step 1: Loading adjacency data...")
        adjacency_matrix, node_ids = self.load_adjacency_data()
        
        #print("Step 2: Performing granular ball clustering...")
        cluster_labels = self.granular_ball_clustering(adjacency_matrix)
        
        #print("Step 3: Creating mapping matrix...")
        mapping_matrix = self.create_mapping_matrix(cluster_labels)
        
        #print(f"Mapping matrix shape: {mapping_matrix.shape}")
        cluster_dist = np.bincount(cluster_labels)
        #print(f"Cluster distribution: {cluster_dist}")
        
        return mapping_matrix, cluster_labels, node_ids

    
    def aggregate_to_coarse(self, fine_data):
    
        if self.mapping_matrix is None:
            raise ValueError("Please perform clustering first!")
        
        print(f"Aggregating: fine_data shape={fine_data.shape}, mapping_matrix shape={self.mapping_matrix.shape}")
        
        if len(fine_data.shape) == 1:
            # 一维数据: (N,) -> (C,)
            return self.mapping_matrix.T @ fine_data
        elif len(fine_data.shape) == 2:
            # 二维数据: (N, T) -> (C, T)
            return self.mapping_matrix.T @ fine_data
        else:
            # 三维数据: (B, N, T) -> (B, C, T)
            # 修复 einsum 操作
            # 原代码: return np.einsum('cn,bnt->bct', self.mapping_matrix.T, fine_data)
            # 应该改为:
            return np.einsum('cn,bnt->bct', self.mapping_matrix.T, fine_data, optimize=True)
    
    def map_to_fine(self, coarse_data):
        
        if self.mapping_matrix is None:
            raise ValueError("Please perform clustering first!")
        
        print(f"Mapping: coarse_data shape={coarse_data.shape}, mapping_matrix shape={self.mapping_matrix.shape}")
        
        if len(coarse_data.shape) == 1:
            # 一维数据: (C,) -> (N,)
            return self.mapping_matrix @ coarse_data
        elif len(coarse_data.shape) == 2:
            # 二维数据: (C, T) -> (N, T)
            return self.mapping_matrix @ coarse_data
        else:
            # 三维数据: (B, C, T) -> (B, N, T)
            # 修复 einsum 操作
            # 原代码: return np.einsum('nc,bcd->bnd', self.mapping_matrix, coarse_data)
            # 应该改为:
            return np.einsum('nc,bct->bnt', self.mapping_matrix, coarse_data, optimize=True)


class TemporalAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        
        # 重塑为 [B*C, T, 1] 然后扩展
        x_flat = x.reshape(B * C, T, 1)
        x_expanded = x_flat.expand(-1, -1, self.input_dim)
        
        Q = self.query(x_expanded)  # [B*C, T, input_dim]
        K = self.key(x_expanded)    # [B*C, T, input_dim]
        V = self.value(x_expanded)  # [B*C, T, input_dim]
        
        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.input_dim)
        attention_weights = self.softmax(attention_scores)
        
        # 应用注意力
        output = torch.matmul(attention_weights, V)  # [B*C, T, input_dim]
        
        # 重塑回原始形状并取均值
        output = output.mean(dim=-1).reshape(B, C, T)
        
        return output

        
class SpatialTemporalFeatureExactor:
    def __init__(self, spatial_processor, input_dim, hidden_dim=64, 
                 output_dim=32, chebyshev_order=3):
        self.spatial_processor = spatial_processor
        self.chebyshev_order = chebyshev_order
        
        # 细粒度特征提取（切比雪夫GCN）
        self.fine_gcn = ChebyshevGCN(input_dim, hidden_dim, output_dim, chebyshev_order)
        
        # 粗粒度特征提取
        self.coarse_gcn = ChebyshevGCN(input_dim, hidden_dim, output_dim, chebyshev_order)
        
        # 特征融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.temporal_attention = TemporalAttention(input_dim)
        self.input_dim = input_dim

    def extract_features(self, fine_window_data):
        print(f"Input to extract_features: {fine_window_data.shape}")
        
        B, N, T = fine_window_data.shape
        
        # 确保已经进行了聚类
        if self.spatial_processor.mapping_matrix is None:
            self.spatial_processor.perform_clustering()
        
        # 获取拉普拉斯矩阵
        L_fine = self.spatial_processor.get_laplacian()
        L_coarse = self.spatial_processor.get_reduced_laplacian()
        
        # 转换为tensor
        fine_tensor = torch.FloatTensor(fine_window_data)
        L_fine_tensor = torch.FloatTensor(L_fine)
        L_coarse_tensor = torch.FloatTensor(L_coarse)
        
        # 1. 细粒度特征提取 [B, N, T] -> [B, N, output_dim]
        fine_features = self.fine_gcn(fine_tensor, L_fine_tensor)
        print(f"After fine GCN: {fine_features.shape}")
        
        # 2. 区域聚合 [B, N, T] -> [B, C, T]
        coarse_window = self.spatial_processor.aggregate_to_coarse(fine_window_data)
        coarse_tensor = torch.FloatTensor(coarse_window)
        print(f"After aggregation: {coarse_tensor.shape}")
        
        # 3. 粗粒度特征提取 [B, C, T] -> [B, C, output_dim]
        coarse_features = self.coarse_gcn(coarse_tensor, L_coarse_tensor)
        print(f"After coarse GCN: {coarse_features.shape}")
        
        # 4. 映射粗粒度特征回细粒度 [B, C, output_dim] -> [B, N, output_dim]
        coarse_mapped = self.spatial_processor.map_to_fine(coarse_features.detach().numpy())
        coarse_mapped_tensor = torch.FloatTensor(coarse_mapped)
        print(f"After mapping coarse to fine: {coarse_mapped_tensor.shape}")
        
        # 5. 时间注意力 [B, N, T] -> [B, N, T]
        temporal_features = self.temporal_attention(fine_tensor)
        print(f"After temporal attention: {temporal_features.shape}")
        
        # 6. 融合细粒度和粗粒度特征
        combined_features = torch.cat([fine_features, coarse_mapped_tensor], dim=-1)
        fused_features = self.fusion_mlp(combined_features)
        print(f"After fusion: {fused_features.shape}")
        
        # 7. 与时间特征进一步融合
        final_features = fused_features + temporal_features.mean(dim=-1, keepdim=True)
        
        return final_features.detach().numpy()
