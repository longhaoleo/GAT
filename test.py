import sys
sys.path.append("src")
import os
import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 导入模块中的函数及类
from GNN.sample import (
    GeneGAT,
    create_full_connected_edge_index,
    create_correlation_edge_index,
    create_knn_edge_index,
    train_graph_classifier,
    extract_pooling_features,
    predict_with_loader,
    save_predictions
)
import gnns.data
from torch_geometric.data import Data, DataLoader
import shap
import xgboost as xgb
from sklearn.model_selection import ParameterSampler


# 设置计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 创建图结构
def create_graph(X_train, method='knn'):
    print(f"\n使用图构建方法: {method}")
    if method == 'full_connected':
        edge_index = create_full_connected_edge_index(X_train.shape[1]).to(device)
    elif method == 'correlation':
        edge_index = create_correlation_edge_index(X_train.cpu().numpy(), threshold=0.6).to(device)
    else:
        edge_index = create_knn_edge_index(X_train.cpu().numpy(), k=15, metric='correlation').to(device)

    print(f"创建的图结构: 边数={edge_index.shape[1]}")
    num_features = X_train.shape[1]
    if edge_index.numel() > 0:
        max_index = edge_index.max().item()
        if num_features <= max_index:
            print(f"警告: 特征数量 ({num_features}) <= 边索引中的最大值 ({max_index}). 调整索引...")
            mask = (edge_index[0] < num_features) & (edge_index[1] < num_features)
            edge_index = edge_index[:, mask]
            print(f"调整后的图结构: 边数={edge_index.shape[1]}")
    return edge_index

# 创建数据加载器
def create_data_objects(X, y, edge_index, batch_size=32):
    data_list = []
    
    # 获取最大节点索引
    max_node_idx = edge_index.max().item() if edge_index.numel() > 0 else 0
    
    for i in range(len(X)):
        x = X[i:i+1].T.float().to(device)
        
        # 确保边索引不超过节点数量
        if edge_index.numel() > 0 and x.size(0) <= max_node_idx:
            # 调整边索引以适应当前样本的节点数量
            valid_edges = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
            if valid_edges.sum() == 0:
                print(f"警告: 样本 {i} 没有有效的边, 使用自循环边.")
                # 创建自循环边
                self_loops = torch.arange(x.size(0)).repeat(2, 1).to(device)
                sample_edge_index = self_loops
            else:
                sample_edge_index = edge_index[:, valid_edges]
                if i == 0:  # 只打印第一个样本的调整信息，避免日志过多
                    print(f"警告: 边索引已调整, 保留了 {valid_edges.sum().item()}/{edge_index.shape[1]} 条边.")
        else:
            sample_edge_index = edge_index
            
        data = Data(x=x, edge_index=sample_edge_index, y=y[i:i+1].to(device))
        data_list.append(data)
    
    if len(data_list) == 0:
        print("错误: 没有有效的数据对象被创建!")
        return None
        
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"创建了数据加载器，包含 {len(data_list)} 个样本，批次大小为 {batch_size}")
    return loader

# 加载和预处理数据
def prepare_data(fold_idx=None, k_fold=5):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    
    # 加载原始数据
    X, y, gene_names, sample_ids = gnns.data.load_data('data', method='lxs', read=True, use_deg=True)
    
    # 标准化特征
    X = StandardScaler().fit_transform(X)
    
    # 转换为张量
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    
    # 如果k_fold=1或不需要K折交叉验证，使用简单的训练/验证/测试分割
    if k_fold == 1 or fold_idx is None:
        # 使用默认划分比例
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
        n_samples = len(X)
        
        # 打乱索引
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        # 计算各集合大小
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        # 划分索引
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
    else:
        # 使用StratifiedKFold进行K折交叉验证
        if fold_idx >= k_fold:
            raise ValueError(f"fold_idx 必须小于 k_fold, 当前 fold_idx={fold_idx}, k_fold={k_fold}")
        
        # 创建K折划分器
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
        
        # 获取所有折的索引
        folds = list(skf.split(X, y))
        
        # 当前折的训练和测试索引
        train_val_indices, test_indices = folds[fold_idx]
        
        # 从训练+验证集中再划分出验证集
        val_size = len(train_val_indices) // 5  # 验证集大小为训练+验证集的20%
        
        # 打乱训练+验证索引
        np.random.seed(42 + fold_idx)  # 不同折使用不同的随机种子
        np.random.shuffle(train_val_indices)
        
        # 划分训练集和验证集
        train_indices = train_val_indices[val_size:]
        val_indices = train_val_indices[:val_size]
    
    # 提取各集合数据
    X_train = X_tensor[train_indices]
    y_train = y_tensor[train_indices]
    X_val = X_tensor[val_indices]
    y_val = y_tensor[val_indices]
    X_test = X_tensor[test_indices]
    y_test = y_tensor[test_indices]
    
    # 创建掩码（仅用于兼容旧代码，实际上不再使用）
    train_mask = torch.zeros(len(X_tensor), dtype=torch.bool)
    val_mask = torch.zeros(len(X_tensor), dtype=torch.bool)
    test_mask = torch.zeros(len(X_tensor), dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    print(f"数据集划分 (Fold {fold_idx if fold_idx is not None else 'Default'})：")
    print(f"  训练样本数: {len(X_train)}")
    print(f"  验证样本数: {len(X_val)}")
    print(f"  测试样本数: {len(X_test)}")
    
    return X_tensor, y_tensor, X_train, y_train, X_val, y_val, X_test, y_test, gene_names, sample_ids, train_mask, val_mask, test_mask

# XGBoost训练或参数搜索
def xgboost_train_or_search(
    X_train, y_train, X_val, y_val, X_test, y_test,
    user_params=None, param_dist=None,
    n_iter=20, random_state=42):
    """
    简洁版：
    - 只用 user_params 时：直接训练并评估
    - 否则：在 param_dist 上随机搜索，并包含 user_params
    全程静默，使用验证集早停，只打印最终结果
    """
    # 基础配置
    base = {
        'objective': 'binary:logistic' if len(np.unique(y_train)) == 2 else 'multi:softprob',
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'verbosity': 0,
        'random_state': random_state
    }

    # 仅直接训练自定义参数
    if user_params and not param_dist:
        cfg = {**base, **user_params}
        model = xgb.XGBClassifier(**cfg)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        try:
            y_probs = model.predict_proba(X_test)
            auc_score = roc_auc_score(y_test, y_probs[:, 1]) if y_probs.shape[1] == 2 else None
        except:
            auc_score = None
            
        metrics = {'accuracy': acc, 'f1_score': f1, 'auc': auc_score}
        print("使用自定义参数：", cfg)
        print(f"测试集评估 -> Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_score if auc_score else 'N/A'}")
        return model, cfg, metrics

    # 构建参数候选列表
    candidates = []
    if user_params:
        candidates.append(user_params)
    if param_dist:
        candidates += list(ParameterSampler(param_dist, n_iter=n_iter, random_state=random_state))

    best_score, best_cfg, best_model = 0, None, None
    # 随机搜索
    for params in candidates:
        cfg = {**base, **params}
        model = xgb.XGBClassifier(**cfg)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        score = accuracy_score(y_val, model.predict(X_val))
        if score > best_score:
            best_score, best_cfg, best_model = score, cfg, model

    # 最终评估并打印
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    try:
        y_probs = best_model.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_probs[:, 1]) if y_probs.shape[1] == 2 else None
    except:
        auc_score = None
        
    metrics = {'accuracy': test_acc, 'f1_score': f1, 'auc': auc_score}
    print("最优参数：", best_cfg)
    print("验证集准确率：", best_score)
    print(f"测试集评估 -> Accuracy: {test_acc:.4f}, F1: {f1:.4f}, AUC: {auc_score if auc_score else 'N/A'}")
    return best_model, best_cfg, metrics

# SHAP分析函数
def perform_shap_analysis(model, X, feature_names=None, output_dir='images/shap'):
    """执行SHAP分析并保存可视化结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n执行SHAP分析...")
    
    # 确保X是NumPy数组
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    
    # 移除了过滤全零特征列的逻辑，直接使用原始特征
    X_filtered = X
    feature_names_filtered = feature_names
    
    # 创建SHAP解释器
    if isinstance(model, xgb.XGBClassifier):
        try:
            # XGBoost模型
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_filtered)
            
            # 对于二分类问题，shap_values是一个列表，我们取第1个元素（正类的SHAP值）
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
                
            # 保存SHAP摘要图
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_filtered, feature_names=feature_names_filtered, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 保存SHAP条形图
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_filtered, feature_names=feature_names_filtered, plot_type='bar', show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_bar.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 保存SHAP依赖图（对于前几个重要特征）
            mean_abs_shap = np.abs(shap_values).mean(0)
            top_indices = np.argsort(mean_abs_shap)[-5:]  # 取前5个重要特征
            
            for idx in top_indices:
                feature_name = feature_names_filtered[idx] if feature_names_filtered is not None else f"Feature {idx}"
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(idx, shap_values, X_filtered, feature_names=feature_names_filtered, show=False)
                plt.title(f"SHAP Dependence Plot - {feature_name}")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shap_dependence_{feature_name}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
            # 保存SHAP值到CSV
            shap_df = pd.DataFrame(shap_values, columns=feature_names_filtered if feature_names_filtered is not None else None)
            shap_df['abs_mean'] = shap_df.abs().mean(axis=0)
            shap_df = shap_df.sort_values('abs_mean', ascending=False)
            shap_df.to_csv(os.path.join(output_dir, 'shap_values.csv'), index=False)
            
            # 保存完整特征重要性到CSV
            full_importance = mean_abs_shap
            full_importance_df = pd.DataFrame({
                'Feature': feature_names if feature_names is not None else [f"Feature_{i}" for i in range(len(full_importance))],
                'Importance': full_importance
            })
            full_importance_df = full_importance_df.sort_values('Importance', ascending=False)
            full_importance_df.to_csv(os.path.join(output_dir, 'full_feature_importance.csv'), index=False)
            
            print(f"SHAP分析完成，结果保存在 {output_dir} 目录")
            return shap_values, full_importance
        except Exception as e:
            print(f"SHAP分析失败: {str(e)}")
            return None, None
    else:
        print("当前模型类型不支持SHAP分析")
        return None, None

# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GeneGAT 训练与推理脚本')
    # parser.add_argument('--load_model', type=str, default='models/k_fold/model_fold_0.pt', help='指定已保存模型的路径, 若提供则直接加载并跳过训练')
    parser.add_argument('--load_model', type=str, default='', help='指定已保存模型的路径, 若提供则直接加载并跳过训练')
    parser.add_argument('--image_dir', type=str, default='images/k_fold', help='保存图片的目录')
    parser.add_argument('--model_dir', type=str, default='models/k_fold', help='保存模型的目录')
    parser.add_argument('--k_fold', type=int, default=1, help='交叉验证折数')
    parser.add_argument('--graph_method', type=str, default='full_connected', 
                        choices=['full_connected', 'correlation', 'knn'], help='图构建方法')
    args, unknown = parser.parse_known_args()

    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 定义XGBoost超参数搜索空间
    param_dist = {
        'n_estimators':    [50, 100, 150, 200],
        'max_depth':       [3, 5, 7, 9],
        'learning_rate':   [0.01, 0.05, 0.1],
        'subsample':       [0.6, 0.8, 1.0],
        'colsample_bytree':[0.6, 0.8, 1.0],
        'gamma':           [0, 0.1, 0.2],
    }
    
    user_params =  {'objective': 'binary:logistic', 'use_label_encoder': False, 'eval_metric': 'logloss', 'verbosity': 0, 'random_state': 42, 'subsample': 1.0, 'n_estimators': 100, 'max_depth': 9, 'learning_rate': 0.1, 'gamma': 0.2, 'colsample_bytree': 0.6}

    # 检查是否需要加载已保存的模型
    if args.load_model and os.path.exists(args.load_model):
        print(f"\n{'='*20} 加载已保存的模型: {args.load_model} {'='*20}")
        
        # 初始化结果存储变量
        fold_results = {
            'gnn': {'accuracy': [], 'f1_score': [], 'auc': []},
            'raw': {'accuracy': [], 'f1_score': [], 'auc': []},
            'graph': {'accuracy': [], 'f1_score': [], 'auc': []},
            'hybrid': {'accuracy': [], 'f1_score': [], 'auc': []}
        }
        
        best_fold = {'fold': 0, 'model_type': None, 'accuracy': 0, 'model': None, 'features': None, 'feature_names': None, 'metrics': None}
        fold_idx = 0  # 设置一个默认的fold_idx值
        
        # 加载数据以获取基因名称和测试集
        X_tensor, y_tensor, X_train, y_train, X_val, y_val, X_test, y_test, gene_names, sample_ids, train_mask, val_mask, test_mask = prepare_data()
        
        # 图构建
        edge_index = create_graph(X_train, method=args.graph_method)
        
        # 加载器创建 - 创建所有必要的数据加载器
        train_loader = create_data_objects(X_train, y_train, edge_index, batch_size=32)
        val_loader = create_data_objects(X_val, y_val, edge_index, batch_size=32)
        test_loader = create_data_objects(X_test, y_test, edge_index, batch_size=32)
        
        # 加载模型
        checkpoint = torch.load(args.load_model)
        
        # 实例化模型
        model = GeneGAT(
            input_dim=1, output_dim=2, hidden_dim=128, pooling_dim=8,
            num_heads=8, gat_layers=2, dropout=0.5,
            activation='swish', use_residual=False, cluster_num=8, 
        ).to(device)
        
        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])
        best_threshold = checkpoint.get('threshold', 0.5)
        
        print(f"模型已加载，使用阈值: {best_threshold}")
        
        # 评估模型
        y_pred, y_probs, features, metrics = predict_with_loader(model, test_loader, device, threshold=best_threshold)
        print(f"GNN评估结果 -> Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, AUC: {metrics['auc']:.4f}")
        
        # 提取池化特征
        print("\n提取池化特征...")
        all_features = {}
        all_features['test'], _ = extract_pooling_features(model, test_loader, device)
        all_features['train'], _ = extract_pooling_features(model, train_loader, device)
        all_features['val'], _ = extract_pooling_features(model, val_loader, device)
        
        # 检查提取的特征是否有效
        for split, features in all_features.items():
            # 确保特征是NumPy数组
            if isinstance(features, torch.Tensor):
                features_np = features.cpu().numpy()
            else:
                features_np = features
                
            zero_cols = np.where(np.all(features_np == 0, axis=0))[0]
            if len(zero_cols) > 0:
                print(f"警告: {split}集中发现 {len(zero_cols)} 个全为0的特征列，索引为: {zero_cols}")
                print(f"这可能表明模型的某些部分没有正确激活或特征提取过程有问题")
        
        all_labels = {}
        all_labels['test'] = y_test
        all_labels['train'] = y_train
        all_labels['val'] = y_val
        
        # 使用不同特征类型训练XGBoost
        feature_types = ['raw', 'graph', 'hybrid']
        for feature_type in feature_types:
            print(f"\n使用 {feature_type} 特征训练XGBoost...")
            
            if feature_type == 'raw':
                X_tr, y_tr = X_train, y_train
                X_va, y_va = X_val, y_val
                X_te, y_te = X_test, y_test
                feature_names = gene_names if gene_names is not None else [f"Gene_{i}" for i in range(X_train.shape[1])]
            elif feature_type == 'graph':
                X_tr, y_tr = all_features['train'], all_labels['train']
                X_va, y_va = all_features['val'], all_labels['val']
                X_te, y_te = all_features['test'], all_labels['test']
                # 为图特征提供简单的编号名称
                feature_names = [f"Emb_{i}" for i in range(all_features['train'].shape[1])]
            else:  # hybrid
                X_tr = np.hstack([X_train, all_features['train']])
                y_tr = y_train
                X_va = np.hstack([X_val, all_features['val']])
                y_va = y_val
                X_te = np.hstack([X_test, all_features['test']])
                y_te = y_test
                # 组合原始特征名称和图特征编号
                raw_names = gene_names if gene_names is not None else [f"Gene_{i}" for i in range(X_train.shape[1])]
                emb_names = [f"Emb_{i}" for i in range(all_features['train'].shape[1])]
                feature_names = raw_names + emb_names
            
            # 训练XGBoost
            xgb_model, best_params, xgb_metrics = xgboost_train_or_search(
                X_tr, y_tr, X_va, y_va, X_te, y_te,
                user_params=user_params, param_dist=None,
                n_iter=30, random_state=42
            )
            
            # 记录XGBoost结果
            fold_results[feature_type]['accuracy'].append(xgb_metrics['accuracy'])
            fold_results[feature_type]['f1_score'].append(xgb_metrics['f1_score'])
            if xgb_metrics['auc'] is not None:
                fold_results[feature_type]['auc'].append(xgb_metrics['auc'])
            
            # 更新最佳模型
            if xgb_metrics['accuracy'] > best_fold['accuracy']:
                best_fold['fold'] = fold_idx
                best_fold['model_type'] = feature_type
                best_fold['accuracy'] = xgb_metrics['accuracy']
                best_fold['model'] = xgb_model
                best_fold['features'] = (X_te, y_te)
                best_fold['feature_names'] = feature_names
                best_fold['metrics'] = xgb_metrics
                                    # 在每一折的hybrid模型上执行SHAP分析
            if feature_type == 'hybrid':
                print(f"\n在第{fold_idx+1}折的hybrid模型上执行SHAP分析...") # 中文打印信息
                fold_shap_dir = os.path.join(args.image_dir, f'fold_{fold_idx}_hybrid_shap')
                
                # 执行整体SHAP分析
                shap_values, feature_importance = perform_shap_analysis(
                    xgb_model, X_te, 
                    feature_names=feature_names,
                    output_dir=fold_shap_dir
                )
                
                # 基于整体 SHAP 分析结果计算贡献度
                if feature_importance is not None and feature_names is not None:
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importance # 使用 perform_shap_analysis 返回的平均绝对SHAP值
                    })
                    importance_df = importance_df.sort_values('Importance', ascending=False)

                    # 保存整体特征重要性（包含原始和图特征）
                    overall_importance_path = os.path.join(fold_shap_dir, 'overall_feature_importance.csv')
                    importance_df.to_csv(overall_importance_path, index=False)
                    print(f"整体特征重要性已保存: {overall_importance_path}") # 中文打印信息

                    # 使用正则表达式区分原始特征和图特征
                    raw_features_df = importance_df[importance_df['Feature'].str.contains('Gene_|^(?!Emb_)', regex=True)]
                    graph_features_df = importance_df[importance_df['Feature'].str.contains('Emb_')]

                    # 分别保存原始特征和图特征的重要性
                    raw_features_df.to_csv(os.path.join(fold_shap_dir, 'raw_features_importance.csv'), index=False)
                    graph_features_df.to_csv(os.path.join(fold_shap_dir, 'graph_features_importance.csv'), index=False)
                    print(f"原始特征和图特征的重要性已分别保存。") # 中文打印信息

                    # 分析并打印总体特征贡献
                    raw_total_importance = raw_features_df['Importance'].sum()
                    graph_total_importance = graph_features_df['Importance'].sum()
                    total_importance = raw_total_importance + graph_total_importance

                    # 避免除零错误
                    raw_percentage = (raw_total_importance / total_importance * 100) if total_importance > 0 else 0
                    graph_percentage = (graph_total_importance / total_importance * 100) if total_importance > 0 else 0

                    print(f"特征贡献分析:") # 中文打印信息
                    print(f"  原始特征总贡献: {raw_total_importance:.4f} ({raw_percentage:.2f}%)") # 中文打印信息
                    print(f"  图特征总贡献: {graph_total_importance:.4f} ({graph_percentage:.2f}%)") # 中文打印信息

                    # 绘制并保存特征贡献比例饼图
                    plt.figure(figsize=(8, 8))
                    plt.pie([raw_total_importance, graph_total_importance],
                           labels=['原始特征', '图特征'], # 中文标签
                           autopct='%1.1f%%',
                           startangle=90,
                           colors=['#ff9999','#66b3ff'])
                    plt.title('原始特征与图特征贡献比例') # 中文标题
                    plt.axis('equal')
                    pie_chart_path = os.path.join(fold_shap_dir, 'feature_contribution_pie.png')
                    plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"特征贡献比例饼图已保存: {pie_chart_path}") # 中文打印信息

                    # 绘制整体特征重要性条形图 (显示所有特征，动态调整高度)
                    plt.figure(figsize=(12, max(8, len(importance_df) * 0.2))) # 动态调整高度
                    sns.barplot(x='Importance', y='Feature', data=importance_df) # 使用完整的 importance_df
                    plt.title('整体特征重要性 (所有特征)') 
                    plt.tight_layout()
                    overall_importance_plot_path = os.path.join(fold_shap_dir, 'overall_feature_importance_all.png') # 修改文件名
                    plt.savefig(overall_importance_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"整体特征重要性图 (所有特征) 已保存: {overall_importance_plot_path}") 

                    # 绘制图特征重要性条形图 (Top N Graph Features)
                    if not graph_features_df.empty:
                        plt.figure(figsize=(12, 8))
                        top_n_graph = min(20, len(graph_features_df)) # 显示Top N 图特征
                        sns.barplot(x='Importance', y='Feature', data=graph_features_df.head(top_n_graph))
                        plt.title(f'Top {top_n_graph} 图特征重要性') 
                        plt.tight_layout()
                        graph_importance_plot_path = os.path.join(fold_shap_dir, 'graph_feature_importance_top.png') # 修改文件名
                        plt.savefig(graph_importance_plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Top {top_n_graph} 图特征重要性图已保存: {graph_importance_plot_path}") 
                    else:
                        print("未找到图特征或图特征重要性为零。") 

    else:
        # 如果没有提供模型或模型不存在，则训练新模型
        print(f"\n{'='*20} 训练新模型 {'='*20}")
        
        # 初始化结果存储变量
        fold_results = {
            'gnn': {'accuracy': [], 'f1_score': [], 'auc': []},
            'raw': {'accuracy': [], 'f1_score': [], 'auc': []},
            'graph': {'accuracy': [], 'f1_score': [], 'auc': []},
            'hybrid': {'accuracy': [], 'f1_score': [], 'auc': []}
        }
        
        best_fold = {'fold': -1, 'model_type': None, 'accuracy': 0, 'model': None, 'features': None, 'feature_names': None, 'metrics': None}
        
        # 执行K折交叉验证
        for fold_idx in range(args.k_fold):
            print(f"\n{'='*20} 第 {fold_idx+1}/{args.k_fold} 折 {'='*20}")
            
            # 准备当前折的数据
            X_tensor, y_tensor, X_train, y_train, X_val, y_val, X_test, y_test, gene_names, sample_ids, train_mask, val_mask, test_mask = prepare_data(fold_idx=fold_idx, k_fold=args.k_fold)
            
            # 图构建
            edge_index = create_graph(X_train, method=args.graph_method)
            
            # 创建数据加载器
            train_loader = create_data_objects(X_train, y_train, edge_index, batch_size=32)
            val_loader = create_data_objects(X_val, y_val, edge_index, batch_size=32)
            test_loader = create_data_objects(X_test, y_test, edge_index, batch_size=32)
            
            # 实例化模型
            model = GeneGAT(
                input_dim=1, output_dim=2, hidden_dim=128, pooling_dim=8,
                num_heads=8, gat_layers=2, dropout=0.5,
                activation='swish', cluster_num=8, 
            ).to(device)
            
            # 训练模型
            print("\n开始训练GNN模型...")
            model, best_threshold = train_graph_classifier(
                model, train_loader, val_loader, 
                epochs=300, patience=100,  class_weights=[1,3],
                lr=1e-4, weight_decay=1e-4,
            )
            
            # 保存模型
            model_path = os.path.join(args.model_dir, f'model_fold_{fold_idx}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'threshold': best_threshold,
            }, model_path)
            print(f"模型已保存: {model_path}")
            
            # 评估模型
            y_pred, y_probs, features, metrics = predict_with_loader(model, test_loader, device, threshold=best_threshold)
            print(f"GNN评估结果 -> Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, AUC: {metrics['auc']:.4f}")
            
            # 记录GNN结果
            fold_results['gnn']['accuracy'].append(metrics['accuracy'])
            fold_results['gnn']['f1_score'].append(metrics['f1_score'])
            if metrics['auc'] is not None:
                fold_results['gnn']['auc'].append(metrics['auc'])
            
            # 提取池化特征
            print("\n提取池化特征...")
            all_features = {}
            all_features['test'], _ = extract_pooling_features(model, test_loader, device)
            all_features['train'], _ = extract_pooling_features(model, train_loader, device)
            all_features['val'], _ = extract_pooling_features(model, val_loader, device)
            
            # 检查提取的特征是否有效
            for split, features in all_features.items():
                # 确保特征是NumPy数组
                if isinstance(features, torch.Tensor):
                    features_np = features.cpu().numpy()
                else:
                    features_np = features
                    
                zero_cols = np.where(np.all(features_np == 0, axis=0))[0]
                if len(zero_cols) > 0:
                    print(f"警告: {split}集中发现 {len(zero_cols)} 个全为0的特征列，索引为: {zero_cols}")
                    print(f"这可能表明模型的某些部分没有正确激活或特征提取过程有问题")
            
            all_labels = {}
            all_labels['test'] = y_test
            all_labels['train'] = y_train
            all_labels['val'] = y_val
            
            # 使用不同特征类型训练XGBoost
            feature_types = ['raw', 'graph', 'hybrid']
            for feature_type in feature_types:
                print(f"\n使用 {feature_type} 特征训练XGBoost...")
                
                if feature_type == 'raw':
                    X_tr, y_tr = X_train, y_train
                    X_va, y_va = X_val, y_val
                    X_te, y_te = X_test, y_test
                    feature_names = gene_names if gene_names is not None else [f"Gene_{i}" for i in range(X_train.shape[1])]
                elif feature_type == 'graph':
                    X_tr, y_tr = all_features['train'], all_labels['train']
                    X_va, y_va = all_features['val'], all_labels['val']
                    X_te, y_te = all_features['test'], all_labels['test']
                    # 为图特征提供简单的编号名称
                    feature_names = [f"Emb_{i}" for i in range(all_features['train'].shape[1])]
                else:  # hybrid
                    X_tr = np.hstack([X_train, all_features['train']])
                    y_tr = y_train
                    X_va = np.hstack([X_val, all_features['val']])
                    y_va = y_val
                    X_te = np.hstack([X_test, all_features['test']])
                    y_te = y_test
                    # 组合原始特征名称和图特征编号
                    raw_names = gene_names if gene_names is not None else [f"Gene_{i}" for i in range(X_train.shape[1])]
                    emb_names = [f"Emb_{i}" for i in range(all_features['train'].shape[1])]
                    feature_names = raw_names + emb_names
                
                # 训练XGBoost
                xgb_model, best_params, xgb_metrics = xgboost_train_or_search(
                    X_tr, y_tr, X_va, y_va, X_te, y_te,
                    user_params=user_params, param_dist=None,
                    n_iter=30, random_state=42
                )
                
                # 记录XGBoost结果
                fold_results[feature_type]['accuracy'].append(xgb_metrics['accuracy'])
                fold_results[feature_type]['f1_score'].append(xgb_metrics['f1_score'])
                if xgb_metrics['auc'] is not None:
                    fold_results[feature_type]['auc'].append(xgb_metrics['auc'])
                
                # 更新最佳模型
                if xgb_metrics['accuracy'] > best_fold['accuracy']:
                    best_fold['fold'] = fold_idx
                    best_fold['model_type'] = feature_type
                    best_fold['accuracy'] = xgb_metrics['accuracy']
                    best_fold['model'] = xgb_model
                    best_fold['features'] = (X_te, y_te)
                    best_fold['feature_names'] = feature_names
                    best_fold['metrics'] = xgb_metrics


