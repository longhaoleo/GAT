"""
基因图构建模块。

data.py 负责把原始表达数据整理成 train/val/test；
这个文件负责把整理好的样本矩阵变成 PyTorch Geometric 图数据。

核心设定：
- 所有样本共用同一套基因关系图 edge_index
- 每个样本的区别在于节点特征，也就是该样本的基因表达量
"""

import torch
import numpy as np
from torch_geometric.data import Data, DataLoader


def get_device():
    """优先使用 GPU；没有 CUDA 时自动回退到 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_full_connected_edge_index(num_nodes, self_loops=True):
    """创建全连接基因图。

    num_nodes 是基因数量。全连接表示所有基因两两相连，让 GAT 自己通过注意力
    学习哪些基因关系更重要。
    """
    if num_nodes <= 0:
        return torch.zeros((2, 0), dtype=torch.long)

    node_indices = torch.arange(num_nodes, dtype=torch.long)
    rows, cols = torch.meshgrid(node_indices, node_indices)
    edge_index = torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=0)

    if not self_loops:
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    # 保留 i <= j，避免无向边重复出现。
    edge_index = edge_index[:, edge_index[0] <= edge_index[1]]
    return edge_index


def create_correlation_edge_index(X, threshold=0.6, self_loops=True):
    """创建相关性基因图。

    X 的形状是 [样本数, 基因数]。这里计算基因列之间的相关性，
    只保留绝对相关系数大于 threshold 的基因对。
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.nan_to_num(corr_matrix)
    mask = np.abs(corr_matrix) >= threshold

    if not self_loops:
        np.fill_diagonal(mask, False)

    rows, cols = np.where(mask)
    if len(rows) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)


def create_knn_edge_index(X, k=10, self_loops=True, metric="correlation"):
    """创建 KNN 基因图。

    对每个基因，找到表达模式最相近的 k 个基因并连边。
    默认使用相关性距离：1 - abs(correlation)。
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    num_genes = X.shape[1]
    effective_k = min(k, num_genes - 1) + (1 if self_loops else 0)
    if effective_k <= 0:
        return torch.zeros((2, 0), dtype=torch.long)

    if metric == "correlation":
        corr_matrix = np.corrcoef(X.T)
        corr_matrix = np.nan_to_num(corr_matrix)
        distance_matrix = 1 - np.abs(corr_matrix)
    else:
        X_t = X.T
        distance_matrix = np.zeros((num_genes, num_genes))
        for i in range(num_genes):
            for j in range(num_genes):
                distance_matrix[i, j] = np.linalg.norm(X_t[i] - X_t[j])

    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=effective_k, metric="precomputed").fit(distance_matrix)
    _, indices = nbrs.kneighbors(distance_matrix)

    source_nodes = np.repeat(np.arange(num_genes), indices.shape[1])
    target_nodes = indices.flatten()

    valid_mask = (target_nodes >= 0) & (target_nodes < num_genes)
    source_nodes = source_nodes[valid_mask]
    target_nodes = target_nodes[valid_mask]

    if not self_loops:
        non_self_mask = source_nodes != target_nodes
        source_nodes = source_nodes[non_self_mask]
        target_nodes = target_nodes[non_self_mask]

    return torch.tensor(np.stack([source_nodes, target_nodes], axis=0), dtype=torch.long)


def create_graph(X_train, method="knn", device=None):
    """基于训练集构建基因之间的边。

    参数说明：
    - full_connected：所有基因两两相连，最简单，但边最多。
    - correlation：表达相关性超过阈值的基因相连。
    - knn：每个基因连接表达模式最相近的 k 个基因。

    只用训练集构图，是为了避免测试集信息泄漏。
    """
    device = device or get_device()
    print(f"\n使用图构建方法: {method}")

    if method == "full_connected":
        edge_index = create_full_connected_edge_index(X_train.shape[1]).to(device)
    elif method == "correlation":
        edge_index = create_correlation_edge_index(X_train.cpu().numpy(), threshold=0.6).to(device)
    else:
        edge_index = create_knn_edge_index(X_train.cpu().numpy(), k=15, metric="correlation").to(device)

    print(f"创建的图结构: 边数={edge_index.shape[1]}")
    num_features = X_train.shape[1]

    # 防御性检查：确保边里的节点编号不会超过当前基因数量。
    if edge_index.numel() > 0:
        max_index = edge_index.max().item()
        if num_features <= max_index:
            print(f"警告: 特征数量 ({num_features}) <= 边索引中的最大值 ({max_index}). 调整索引...")
            mask = (edge_index[0] < num_features) & (edge_index[1] < num_features)
            edge_index = edge_index[:, mask]
            print(f"调整后的图结构: 边数={edge_index.shape[1]}")

    return edge_index


def create_data_objects(X, y, edge_index, batch_size=32, device=None):
    """把样本矩阵转换成 PyG 的 DataLoader。

    X 的形状是 [样本数, 基因数]。对第 i 个样本：
    - 取 X[i] 得到该样本所有基因表达值
    - 转置成 [基因数, 1]，让每个基因成为一个节点
    - 所有样本复用同一个 edge_index，也就是同一套基因关系图

    换句话说：所有样本共用一个基因图，区别只在节点表达量和标签。
    """
    device = device or get_device()
    data_list = []
    max_node_idx = edge_index.max().item() if edge_index.numel() > 0 else 0

    for i in range(len(X)):
        # 单个样本的表达向量转成节点特征矩阵：[num_genes, 1]。
        x = X[i : i + 1].T.float().to(device)

        # 防御性编程：确保边索引不会越界。
        if edge_index.numel() > 0 and x.size(0) <= max_node_idx:
            valid_edges = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
            if valid_edges.sum() == 0:
                print(f"警告: 样本 {i} 没有有效的边, 使用自循环边.")
                sample_edge_index = torch.arange(x.size(0)).repeat(2, 1).to(device)
            else:
                sample_edge_index = edge_index[:, valid_edges]
                if i == 0:
                    print(f"警告: 边索引已调整, 保留了 {valid_edges.sum().item()}/{edge_index.shape[1]} 条边.")
        else:
            sample_edge_index = edge_index

        # Data 表示一张样本图。sample_edge_index 是共享的基因关系，
        # x 是当前样本自己的基因表达量，y 是当前样本的早/晚期标签。
        data = Data(x=x, edge_index=sample_edge_index, y=y[i : i + 1].to(device))
        data_list.append(data)

    if len(data_list) == 0:
        print("错误: 没有有效的数据对象被创建!")
        return None

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"创建了数据加载器，包含 {len(data_list)} 个样本，批次大小为 {batch_size}")
    return loader
