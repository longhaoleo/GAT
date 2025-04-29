# 基因为节点 简单版
import sys
sys.path.append("src")
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler
# 导入模块中的函数及类
from GNN.sample import GeneGAT, create_full_connected_edge_index, create_correlation_edge_index, create_knn_edge_index, train_graph_classifier, create_data_loaders, predict_with_loader, save_predictions,extract_pooling_features
import gnns.data
from torch_geometric.data import Data, Batch, DataLoader 

# 设置计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 加载和预处理数据
def prepare_data():
    # 加载数据
    X, y, gene_names, sample_ids = gnns.data.load_data('data', method='lxs', read=False, use_deg=True)
    
    # 划分数据集并标准化
    X, y, train_mask, val_mask, test_mask = gnns.data.get_split_data(X, y, oversample=True)
    X = StandardScaler().fit_transform(X)
    global X_tensor
    # 转换为PyTorch张量
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    train_mask = torch.from_numpy(train_mask)
    val_mask = torch.from_numpy(val_mask)
    test_mask = torch.from_numpy(test_mask)
    # 提取训练/验证/测试数据
    X_train = X_tensor[train_mask][:100]
    y_train = y_tensor[train_mask][:100]
    X_val = X_tensor[val_mask]
    y_val = y_tensor[val_mask]
    X_test = X_tensor[test_mask]
    y_test = y_tensor[test_mask]
    
    print("数据集划分：")
    print(f"  训练样本数: {len(X_train)}")
    print(f"  验证样本数: {len(X_val)}")
    print(f"  测试样本数: {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, sample_ids, test_mask

# 创建图结构
def create_graph(X_train, method='knn'):
    print(f"\n使用图构建方法: {method}")
    
    if method == 'full_connected':
        edge_index = create_full_connected_edge_index(X_train.shape[1]).to(device)
    elif method == 'correlation':
        X_train_genes = X_train.cpu().numpy()
        edge_index = create_correlation_edge_index(X_train_genes, threshold=0.6).to(device)
    elif method == 'knn':
        X_train_genes = X_train.cpu().numpy()
        edge_index = create_knn_edge_index(X_train_genes, k=25, metric='correlation').to(device)
    
    print(f"创建的图结构: 边数={edge_index.shape[1]}")
    
    # 检查边索引是否超出节点数量范围
    num_features = X_train.shape[1]
    if edge_index.numel() > 0:
        max_index = edge_index.max().item()
        if num_features <= max_index:
            print(f"警告: 特征数量 ({num_features}) 小于或等于边索引中的最大值 ({max_index}).")
            print("正在调整边索引以匹配特征数量...")
            edge_index = edge_index[:, edge_index[0] < num_features]
            edge_index = edge_index[:, edge_index[1] < num_features]
            print(f"调整后的图结构: 边数={edge_index.shape[1]}")
    
    return edge_index

# 创建数据加载器
def create_data_objects(X_train, y_train, X_val, y_val, X_test, y_test, edge_index, batch_size=256):
    train_data_list = []
    val_data_list = []
    test_data_list = []
    
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
                print(f"警告: 样本 {i} 的节点数 ({num_nodes}) 小于或等于边索引中的最大值 ({max_index}).")
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
    
    # 为测试集创建数据对象
    for i in range(len(X_test)):
        x = X_test[i:i+1].T.float().to(device)
        y = y_test[i:i+1].to(device)
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # 检查数据对象是否有效
        if data.edge_index.numel() > 0:
            max_index = torch.max(data.edge_index)
            num_nodes = data.x.size(0)
            if num_nodes <= max_index:
                print(f"警告: 测试样本 {i} 的节点数 ({num_nodes}) 小于或等于边索引中的最大值 ({max_index}).")
                continue
        test_data_list.append(data)
    
    # 创建数据加载器
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

# 主函数
def main():
    # 创建目录
    single_image_dir = r'c:\Users\llh17\Desktop\GNNs\image\single_run'
    single_model_dir = r'c:\Users\llh17\Desktop\GNNs\models\single_run'
    os.makedirs(single_image_dir, exist_ok=True)
    os.makedirs(single_model_dir, exist_ok=True)
    global X_tensor,X_train, y_train, X_val, y_val, X_test, y_test
    global model
    global edge_index
    global train_loader, val_loader, test_loader
    global best_threshold
    # 准备数据
    X_train, y_train, X_val, y_val, X_test, y_test, sample_ids, test_mask = prepare_data()
    
    # 创建模型
    model = GeneGAT(
        input_dim=1,
        output_dim=2,
        hidden_dim=64,
        pooling_dim=16,
        num_heads=4, 
        gat_layers=2,
        dropout=0.5,
        activation='swish',
        cluster_num=8,
        key_node_ratio=0.2,
    ).to(device)
    
    # 创建图结构
    graph_method = 'knn'
    edge_index = create_graph(X_tensor, graph_method)
    # edge_index = create_graph(X_train, graph_method)
    # 创建数据加载器
    batch_size = 32
    train_loader, val_loader, test_loader = create_data_objects(
        X_train, y_train, X_val, y_val, X_test, y_test, edge_index, batch_size)
    
    # 训练模型
    print("\n开始训练模型...")
    model, best_threshold = train_graph_classifier(
        model=model, 
        class_weights=[1,2],
        train_data=train_loader,
        val_data=val_loader,
        epochs=200, 
        lr=0.0001,
        weight_decay=1e-4,
        patience=100,
    )
    
    # 使用最佳阈值进行预测
    y_pred, y_probs, features, metrics = predict_with_loader(model, test_loader, device, threshold=best_threshold)
    
    # 提取池化后降维后的特征
    print("\n提取池化后降维后的特征...")
    pooled_features, pool_labels = extract_pooling_features(
        model=model,
        data_loader=test_loader,
        device=device
    )
    
    # 可视化降维后的特征
    if pooled_features is not None:
        print(f"可视化降维后的特征...")
        pool_features = pooled_features.cpu().numpy()
        
        # 使用t-SNE降维
        pool_features_2d = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=min(30, len(pool_features)-1)
        ).fit_transform(pool_features)
        
        # 绘制t-SNE图
        plt.figure(figsize=(10, 8))
        y_test_np = pool_labels.cpu().numpy()
        for i in range(2):
            plt.scatter(
                pool_features_2d[y_test_np == i, 0], 
                pool_features_2d[y_test_np == i, 1], 
                label=f'Class {i}', 
                alpha=0.7
            )
        plt.title(f't-SNE Visualization of Pooled Features')
        plt.legend()
        plt.savefig(os.path.join(single_image_dir, f'pooled_features_visualization.png'))
        plt.close()
    
    # 提取评估指标
    accuracy = metrics['accuracy']
    test_f1 = metrics['f1_score']
    roc_auc = metrics['auc']
    
    # 保存预测结果
    predictions_save_path = os.path.join(r'c:\Users\llh17\Desktop\GNNs\results', 'predictions_single_run.csv')
    
    # 获取测试样本ID
    test_sample_ids = None
    if sample_ids is not None:
        sample_ids_array = np.array(sample_ids)
        test_mask_np = test_mask.cpu().numpy()
        if len(test_mask_np) == len(sample_ids_array):
            test_sample_ids = sample_ids_array[test_mask_np].tolist()
    
    # 保存预测结果
    predictions_df = save_predictions(
        y_true=y_test.cpu(),
        y_pred=y_pred.cpu(),
        y_probs=y_probs.cpu(),
        sample_ids=test_sample_ids,
        features=features.cpu(),
        save_path=predictions_save_path
    )
    
    # 可视化结果
    # t-SNE可视化
    features_np = features.cpu().detach().numpy()
    y_test_np = y_test.cpu().numpy()
    features_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_np)-1)).fit_transform(features_np)
    
    plt.figure(figsize=(10, 8))
    for i in range(2):
        plt.scatter(features_2d[y_test_np == i, 0], features_2d[y_test_np == i, 1], label=f'Class {i}', alpha=0.7)
    plt.title('t-SNE Feature Visualization')
    plt.legend()
    plt.savefig(os.path.join(single_image_dir, 'feature_visualization.png'))
    plt.close()
    
    # 混淆矩阵
    y_pred_np = y_pred.cpu().numpy()
    cm = confusion_matrix(y_test_np, y_pred_np)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(single_image_dir, 'confusion_matrix.png'))
    plt.close()
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test_np, y_probs[:, 1].cpu().numpy())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(single_image_dir, 'roc_curve.png'))
    plt.close()
    
    # 保存模型及结果
    model_save_path = os.path.join(single_model_dir, 'model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': best_threshold,
        'input_dim': 1,
        'hidden_dim': model.hidden_dim,
        'output_dim': 2,
        'accuracy': accuracy,
        'f1_score': test_f1,
        'auc': roc_auc,
        'graph_method': graph_method
    }, model_save_path)
    print(f"\nSaved model: {model_save_path}")
    
    # 保存测试结果
    results = {
        'accuracy': [accuracy],
        'f1_score': [test_f1],
        'auc': [roc_auc],
        'threshold': [best_threshold]
    }
    results_df = pd.DataFrame(results, index=['Test'])
    results_save_path = os.path.join(r'c:\Users\llh17\Desktop\GNNs\results', 'results_single_run.csv')
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
    results_df.to_csv(results_save_path)
    print(f"\nTest results saved to: {results_save_path}")

if __name__ == "__main__":
    main()