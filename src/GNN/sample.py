# 基因为节点，复杂版本
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch.optim import AdamW
import time
import os
import copy
import math
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score  # Add this line
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
import warnings
warnings.filterwarnings("ignore")

# 保持原有的图构建函数，无需偏移调整
import torch

def create_full_connected_edge_index(num_nodes, self_loops=True):
    if num_nodes <= 0:
        return torch.zeros((2, 0), dtype=torch.long)

    node_indices = torch.arange(num_nodes, dtype=torch.long)
    rows, cols = torch.meshgrid(node_indices, node_indices)
    edge_index = torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=0)

    # 如果不需要自环，先去掉自环
    if not self_loops:
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]

    # 去除重复无向边：仅保留 i <= j（或 i < j，若也要去掉自环，就用 <）
    mask = edge_index[0] <= edge_index[1]
    edge_index = edge_index[:, mask]

    return edge_index


def create_correlation_edge_index(X, threshold=0.6, self_loops=True):
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    num_genes = X.shape[1]
    corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.nan_to_num(corr_matrix)
    mask = np.abs(corr_matrix) >= threshold

    if not self_loops:
        np.fill_diagonal(mask, False)

    rows, cols = np.where(mask)
    edges = np.stack([rows, cols], axis=1)

    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index

def create_knn_edge_index(X, k=10, self_loops=True, metric='correlation'):
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    num_genes = X.shape[1]
    effective_k = min(k, num_genes-1) + (1 if self_loops else 0)

    if effective_k <= 0:
        return torch.zeros((2, 0), dtype=torch.long)

    if metric == 'correlation':
        corr_matrix = np.corrcoef(X.T)
        corr_matrix = np.nan_to_num(corr_matrix)
        distance_matrix = 1 - np.abs(corr_matrix)
    else:
        X_t = X.T
        dist_matrix = np.zeros((num_genes, num_genes))
        for i in range(num_genes):
            for j in range(num_genes):
                dist_matrix[i, j] = np.linalg.norm(X_t[i] - X_t[j])
        distance_matrix = dist_matrix

    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=effective_k, metric='precomputed').fit(distance_matrix)
    _, indices = nbrs.kneighbors(distance_matrix)

    source_nodes = np.repeat(np.arange(num_genes), indices.shape[1])
    target_nodes = indices.flatten()

    if np.any(target_nodes >= num_genes) or np.any(target_nodes < 0):
        valid_mask = (target_nodes >= 0) & (target_nodes < num_genes)
        source_nodes = source_nodes[valid_mask]
        target_nodes = target_nodes[valid_mask]

    if not self_loops:
        mask = source_nodes != target_nodes
        source_nodes = source_nodes[mask]
        target_nodes = target_nodes[mask]

    edges = np.stack([source_nodes, target_nodes], axis=0)
    edge_index = torch.tensor(edges, dtype=torch.long)

    return edge_index

def create_data_loaders(train_data, val_data, batch_size=32, num_workers=0):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def train_graph_classifier(model, train_data, val_data=None, batch_size=32, epochs=100, lr=0.001,
                            weight_decay=1e-5, patience=15, class_weights=None, num_workers=0):
    """
    训练图分类模型，支持直接传入数据集或数据加载器

    参数:
        model: PyTorch Geometric模型实例
        train_data: 训练数据集或训练数据加载器
        val_data: 验证数据集或验证数据加载器
        batch_size: 批处理大小（当传入数据集时使用）
        epochs: 最大训练轮数
        lr: 学习率
        weight_decay: AdamW优化器的权重衰减
        patience: 早停耐心值
        class_weights: 可选的类别权重张量（用于FocalLoss/CrossEntropy）
        num_workers: 数据加载的工作线程数

    返回:
        model: 训练好的模型
        best_threshold: 验证集上表现最佳的阈值
    """

    # 检查输入是否为DataLoader，如果不是则创建DataLoader
    if not isinstance(train_data, DataLoader):
        print(f"将数据集转换为DataLoader，批处理大小: {batch_size}")
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        train_loader = train_data
        val_loader = val_data
        print(f"使用提供的DataLoader: 训练批次数={len(train_loader)}, 验证批次数={len(val_loader)}")

    # 使用固定学习率的优化器
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 删除动态学习率调度器
    
    # Setup Loss Function
    if class_weights is not None:
        # Ensure class_weights tensor is created correctly before passing
        alpha=torch.tensor(class_weights, dtype=torch.float)
        print("alpha shape:", alpha.shape)
        print("alpha:", alpha)
        if torch.isnan(alpha).any() or torch.isinf(alpha).any():
            print("Alpha contains NaN or Inf!")
            alpha = torch.nan_to_num(alpha)   # 或直接裁剪 alpha
        criterion = FocalLoss(alpha=alpha, gamma=2)
        print("Using weighted loss.")
    else:
        criterion = FocalLoss(gamma=2) # Or nn.CrossEntropyLoss()
        print("Using unweighted Focal loss.")


    best_model_state = None
    epochs_no_improve = 0
    threshold = 0.5   # 使用固定阈值
    best_f1 = 0.0   # 用于跟踪最佳F1分数
    best_val_loss = float('inf')   # 用于跟踪最佳验证损失
    print(f"Starting training for {epochs} epochs...")
    # 训练循环
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        model.train()
        total_loss = 0
        correct_train = 0
        total_train_samples = 0
        optimizer.zero_grad() # 每个epoch开始时清零梯度

        # --- 训练循环 (遍历批次) ---
        for batch in train_loader:
            batch = batch
            
            # 添加批次检查
            if batch.edge_index.numel() > 0:
                max_index = torch.max(batch.edge_index)
                num_nodes = batch.x.size(0)
                if num_nodes <= max_index:
                    print(f"警告: 批次的节点数 ({num_nodes}) 小于或等于边索引中的最大值 ({max_index}).")
                    continue

            # 前向传播
            logits = model(batch)
            target = batch.y.squeeze().long()

            # 计算损失
            loss = criterion(logits, target)
            
            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            # --- 记录和指标 ---
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct_train += (pred == target).sum().item()
            total_train_samples += batch.num_graphs

        avg_train_loss = total_loss / len(train_loader) # Average loss over batches
        train_acc = correct_train / total_train_samples
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val_samples = 0
        all_val_preds = []    # Initialize the list here
        all_val_targets = []  # Initialize the 


        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch)
                target = batch.y.squeeze().long()
                loss = criterion(logits, target)
                total_val_loss += loss.item()

                # 使用固定阈值进行预测
                probs = torch.softmax(logits, dim=1)
                pred = (probs[:, 1] > threshold).int()
                
                all_val_preds.append(pred)
                all_val_targets.append(target)
                correct_val += (pred == target).sum().item()
                total_val_samples += batch.num_graphs

        # 计算验证指标
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val / total_val_samples

        # 计算F1分数
        all_val_preds = torch.cat(all_val_preds).cpu().numpy()
        all_val_targets = torch.cat(all_val_targets).cpu().numpy()
        current_f1 = f1_score(all_val_targets, all_val_preds, average='binary')

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoch {epoch}/{epochs} [{epoch_duration:.2f}s]: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"Val F1: {current_f1:.4f}")

        # --- Early Stopping ---
        improved = False
        
        # 检查验证损失是否降低
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            improved = True
            print(f"    验证损失提升到 {best_val_loss:.4f}")
            
        # 检查F1分数是否提高
        if current_f1 > best_f1:
            best_f1 = current_f1
            improved = True
            print(f"    F1分数提升到 {best_f1:.4f}")
        
        # 如果有任何改进，保存模型
        if improved:
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"    模型状态已保存。")
        else:
            epochs_no_improve += 1
            print(f"    验证指标未提升 {epochs_no_improve} 轮。")
            if epochs_no_improve >= patience:
                print(f"早停在第 {epoch} 轮触发。")
                break

    # --- Load Best Model ---
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"加载最佳模型状态，验证损失: {best_val_loss:.4f}, F1分数: {best_f1:.4f}")
    else:
        print("训练完成，但没有改进或早停未触发。")

    return model, threshold


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        # 确保alpha正确处理，可能作为张量[alpha_0, alpha_1]
        if isinstance(alpha, (list, tuple, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        else:
            self.alpha = alpha # 也可以是标量

        self.gamma = gamma
        self.reduction = reduction
        # 最初使用reduction='none'以便按元素应用alpha和gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        # 确保targets是长整型
        targets = targets.long()
        # 计算每个元素的交叉熵损失
        ce_loss = self.ce_loss(inputs, targets)
        # 计算pt（真实类别的概率）
        pt = torch.exp(-ce_loss)

        # 计算alpha因子
        if isinstance(self.alpha, torch.Tensor):
            # 确保alpha张量与输入在同一设备上
            alpha_t = self.alpha.to(inputs.device)[targets]
        else:
            # 如果alpha是标量，直接使用
            alpha_t = self.alpha

        # 计算Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        # 应用最终的reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss


# 修改 GeneGAT 类，仅保留簇池化
class GeneGAT(nn.Module):
    def __init__(self, input_dim=1, output_dim=2, num_heads=2, hidden_dim=128, gat_layers=2, dropout=0.1, 
                 activation='relu', use_residual=True, 
                 cluster_num=8, key_node_ratio=0.2,  pooling_dim=16):
        super(GeneGAT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation_fn = self._get_activation(activation)
        self.use_residual = use_residual
        self.cluster_num = cluster_num

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.gat_convs = nn.ModuleList()
        current_dim = hidden_dim
        for i in range(gat_layers):
            is_last_layer = (i == gat_layers - 1)
            concat = not is_last_layer
            out_channels_per_head = hidden_dim // num_heads
            layer_out_dim = out_channels_per_head * num_heads if concat else hidden_dim
            conv_out_channels = out_channels_per_head if concat else hidden_dim

            conv = GATConv(current_dim, conv_out_channels, heads=num_heads, dropout=dropout, concat=concat)
            self.gat_convs.append(conv)
            current_dim = layer_out_dim

        self.final_gat_dim = current_dim

        # 添加残差连接
        if self.use_residual:
            self.res_connections = nn.ModuleList([
                nn.Linear(hidden_dim, self.final_gat_dim)
                for _ in range(len(self.gat_convs))
            ])

        # 簇分配层 - 用于学习节点到簇的软分配 - 移除ReLU激活函数
        self.cluster_assignment = nn.Sequential(
            nn.Linear(self.final_gat_dim, self.final_gat_dim // 2),
            # 移除ReLU激活函数，允许负值传递
            nn.Linear(self.final_gat_dim // 2, cluster_num),
            nn.Softmax(dim=1)  # 保留Softmax以确保分配权重和为1
        )

        # 最终池化后的特征维度是簇池化的输出维度的两倍（最大池化+最小池化）
        self.final_pooled_dim = self.final_gat_dim * 2  # 修改为2倍，因为拼接了最大和最小池化

        # 在计算完final_pooled_dim之后添加降维层
        print(f"池化前特征维度: {self.final_gat_dim}, 拼接后维度: {self.final_pooled_dim}")
        self.pooling_dim = min(pooling_dim, self.final_pooled_dim)  # 确保降维后的维度不大于原维度
        
        # 添加池化后的降维层 - 使用LeakyReLU替代ReLU以保留负值信息
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.final_pooled_dim, self.pooling_dim),
            nn.LayerNorm(self.pooling_dim),
            nn.LeakyReLU(0.1)  # 使用LeakyReLU替代ReLU，允许负值以小斜率传递
        )
        
        # 修改分类器的输入维度
        self.classifier = nn.Sequential(
            nn.Linear(self.pooling_dim, output_dim),
            nn.Dropout(dropout)
        )

        self.reset_parameters()

    def _get_activation(self, name):
        if name == 'relu': return nn.ReLU()
        if name == 'elu': return nn.ELU()
        if name == 'leaky_relu': return nn.LeakyReLU(0.2)
        if name == 'mish': return nn.Mish()
        if name =='swish': return nn.SiLU()
        return nn.ReLU()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        x = x.float()

        # 输入投影、归一化和激活
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.activation_fn(x)
        x_init = x  # 保留初始特征用于残差连接
        x = F.dropout(x, p=self.dropout * 0.5, training=self.training)

        # GAT 卷积层（支持残差连接）
        for i, conv in enumerate(self.gat_convs):
            if self.use_residual:
                x_res = self.res_connections[i](x_init)
                x_conv = conv(x, edge_index)
                x = x_conv + x_res  # 残差连接
            else:
                x = conv(x, edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x_init = x  # 更新残差连接的基准

        # 簇池化
        batch_size = batch.max().item() + 1
        
        # 计算节点到簇的软分配矩阵
        cluster_assign = self.cluster_assignment(x)  # [num_nodes, cluster_num]
        
        # 对每个图分别处理
        cluster_features = []
        for b in range(batch_size):
            # 获取当前图的节点
            mask = (batch == b)
            graph_x = x[mask]  # [graph_nodes, in_dim]
            graph_assign = cluster_assign[mask]  # [graph_nodes, cluster_num]
            
            # 计算簇特征: C = A^T * X
            graph_clusters = torch.matmul(graph_assign.t(), graph_x)  # [cluster_num, in_dim]
            
            # 使用最大最小池化拼接替代均值池化
            max_pooled = graph_clusters.max(dim=0).values  # [in_dim]
            min_pooled = graph_clusters.min(dim=0).values  # [in_dim]
            graph_cluster_repr = torch.cat([max_pooled, min_pooled], dim=0)  # [in_dim*2]
            cluster_features.append(graph_cluster_repr)
        
        graph_embedding = torch.stack(cluster_features)  # [batch_size, in_dim*2]
        
        # 应用降维层
        graph_embedding = self.dim_reduction(graph_embedding)
        
        # 分类器输出
        logits = self.classifier(graph_embedding)
        return logits

    def extract_features(self, data):
        """
        提取图特征（只返回池化后的特征）
        
        参数:
            data: PyG Data 对象
            
        返回:
            graph_embedding: 池化后的图特征
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)

        x = x.float()

        # 输入处理
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.activation_fn(x)
        x_init = x
        x = F.dropout(x, p=self.dropout * 0.5, training=self.training)

        # GAT 卷积层
        for i, conv in enumerate(self.gat_convs):
            if self.use_residual:
                x_res = self.res_connections[i](x_init)
                x_conv = conv(x, edge_index)
                x = x_conv + x_res
            else:
                x = conv(x, edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x_init = x

        # 簇池化
        batch_size = batch.max().item() + 1
        
        # 计算节点到簇的软分配矩阵
        cluster_assign = self.cluster_assignment(x)  # [num_nodes, cluster_num]
        
        # 对每个图分别处理
        cluster_features = []
        for b in range(batch_size):
            # 获取当前图的节点
            mask = (batch == b)
            graph_x = x[mask]  # [graph_nodes, in_dim]
            graph_assign = cluster_assign[mask]  # [graph_nodes, cluster_num]
            
            # 计算簇特征: C = A^T * X
            graph_clusters = torch.matmul(graph_assign.t(), graph_x)  # [cluster_num, in_dim]
            
            # 使用最大最小池化拼接替代均值池化
            max_pooled = graph_clusters.max(dim=0).values  # [in_dim]
            min_pooled = graph_clusters.min(dim=0).values  # [in_dim]
            graph_cluster_repr = torch.cat([max_pooled, min_pooled], dim=0)  # [in_dim*2]
            cluster_features.append(graph_cluster_repr)
        
        graph_embedding = torch.stack(cluster_features)  # [batch_size, in_dim*2]
        
        # 应用降维层
        graph_embedding = self.dim_reduction(graph_embedding)
        
        return graph_embedding



def save_predictions(y_true, y_pred, y_probs, sample_ids=None, features=None, save_path=None):
    """
    保存预测结果到CSV文件

    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    - y_probs: 预测概率
    - sample_ids: 样本ID（可选）
    - features: 特征向量（可选）
    - save_path: 保存路径

    返回:
    - 保存的DataFrame
    """
    # 确保所有输入都是NumPy数组
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    y_probs = y_probs.cpu().numpy() if isinstance(y_probs, torch.Tensor) else y_probs

    # 创建基本结果DataFrame
    results = {
        'true_label': y_true,
        'predicted_label': y_pred,
        'prob_class_0': y_probs[:, 0] if y_probs.ndim > 1 else 1 - y_probs,
        'prob_class_1': y_probs[:, 1] if y_probs.ndim > 1 else y_probs,
        'correct': (y_true == y_pred).astype(int)
    }

    # 如果提供了样本ID，添加到结果中
    if sample_ids is not None:
        results['sample_id'] = sample_ids

    # 创建DataFrame
    results_df = pd.DataFrame(results)

    # 如果提供了特征，添加到结果中
    if features is not None:
        features = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
        feature_cols = {f'feature_{i}': features[:, i] for i in range(features.shape[1])}
        feature_df = pd.DataFrame(feature_cols)
        results_df = pd.concat([results_df, feature_df], axis=1)

    # 保存结果
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        results_df.to_csv(save_path, index=False)
        print(f"预测结果已保存到: {save_path}")

    return results_df

def check_gradients(model, step=0, log_interval=100):
    """检查模型参数的梯度"""
    if step % log_interval != 0:
        return
    
    zero_grad_params = []
    none_grad_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                none_grad_params.append(name)
            elif torch.all(param.grad == 0):
                zero_grad_params.append(name)
    
    if none_grad_params:
        print(f"警告: 以下参数的梯度为None: {none_grad_params}")
    if zero_grad_params:
        print(f"警告: 以下参数的梯度全为0: {zero_grad_params}")


def create_graph_data_loaders(X_train, y_train, X_val, y_val, edge_index, device, batch_size=256):
    """
    为训练集和验证集创建PyG数据加载器
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        X_val: 验证集特征
        y_val: 验证集标签
        edge_index: 图的边索引
        device: 计算设备
        batch_size: 批处理大小
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    train_data_list = []
    val_data_list = []

    # 为训练集创建数据对象
    for i in range(len(X_train)):
        x = X_train[i:i+1].T.float().to(device)
        y = y_train[i:i+1].to(device)
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # 检查数据对象是否有效
        if data.edge_index.numel() > 0:
            max_index = torch.max(data.edge_index)
            num_nodes = data.x.size(0)
            if num_nodes <= max_index:
                print(f"警告: 训练样本 {i} 的节点数 ({num_nodes}) 小于或等于边索引中的最大值 ({max_index}).")
                continue
        train_data_list.append(data)
    
    # 为验证集创建数据对象
    for i in range(len(X_val)):
        x = X_val[i:i+1].T.float().to(device)
        y = y_val[i:i+1].to(device)
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # 检查数据对象是否有效
        if data.edge_index.numel() > 0:
            max_index = torch.max(data.edge_index)
            num_nodes = data.x.size(0)
            if num_nodes <= max_index:
                print(f"警告: 验证样本 {i} 的节点数 ({num_nodes}) 小于或等于边索引中的最大值 ({max_index}).")
                continue
        val_data_list.append(data)

    # 创建数据加载器
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

def predict_with_loader(model, test_loader, device, threshold=0.5):
    """
    使用数据加载器进行预测和评估，使用固定阈值
    
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        threshold: 分类阈值（固定值）
        
    返回:
        total_predictions: 预测标签
        total_probabilities: 预测概率
        all_features: 提取的特征
        metrics: 评估指标字典
    """
    model.eval()
    total_predictions = []
    total_probabilities = []
    total_labels = []
    all_features = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # 提取特征和计算预测
            logits = model(batch)
            features = model.extract_features(batch)
            
            # 计算概率和预测
            probs = torch.softmax(logits, dim=1)
            predictions = (probs[:, 1] > threshold).int()
            
            # 收集结果
            total_predictions.append(predictions.cpu())
            total_probabilities.append(probs.cpu())
            total_labels.append(batch.y.squeeze().cpu())
            all_features.append(features.cpu())

    # 连接所有批次的结果
    total_predictions = torch.cat(total_predictions)
    total_probabilities = torch.cat(total_probabilities)
    total_labels = torch.cat(total_labels)
    all_features = torch.cat(all_features)

    # 转换为NumPy数组
    total_predictions_np = total_predictions.numpy()
    total_probabilities_np = total_probabilities.numpy()
    total_labels_np = total_labels.numpy()

    # 计算评估指标
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(total_labels_np, total_predictions_np)
    f1 = f1_score(total_labels_np, total_predictions_np, average='binary')
    auc_score = roc_auc_score(total_labels_np, total_probabilities_np[:, 1])
    
    print(f"测试集评估结果 (固定阈值={threshold:.4f}):")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  AUC: {auc_score:.4f}")
    
    metrics = {
        'accuracy': accuracy, 
        'f1_score': f1, 
        'auc': auc_score,
        'threshold': threshold
    }

    return total_predictions, total_probabilities, all_features, metrics


def extract_pooling_features(model, data_loader, device):
    """
    从模型中提取池化后降维后的特征
    
    参数:
        model: 训练好的GeneGAT模型
        data_loader: 数据加载器
        device: 计算设备
        
    返回:
        pooled_features: 池化后降维后的特征
        labels: 对应的标签
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # 使用模型的extract_features方法获取池化后降维后的特征
            features = model.extract_features(batch)
            all_features.append(features.cpu())
            
            # 收集标签
            all_labels.append(batch.y.squeeze().cpu())
    
    # 连接所有批次的结果
    all_labels = torch.cat(all_labels)
    
    # 连接特征
    if all_features:
        pooled_features = torch.cat(all_features)
    else:
        pooled_features = None
    
    return pooled_features, all_labels




