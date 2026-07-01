"""
GeneGAT 模型与预测/特征提取工具。

graph.py 负责把表达矩阵变成“基因为节点”的 PyG 图；
这个文件只负责图神经网络本身：
- GeneGAT 模型结构
- 测试集预测
- 提取池化后的图嵌入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from torch_geometric.nn import GATConv

from gnns.metrics import evaluate_classification, print_metrics

import warnings
warnings.filterwarnings("ignore")



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

    metrics = evaluate_classification(
        total_labels_np,
        total_predictions_np,
        y_prob=total_probabilities_np,
        average="binary",
        threshold=threshold,
    )
    print_metrics(f"测试集评估结果 (固定阈值={threshold:.4f})", metrics)

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




