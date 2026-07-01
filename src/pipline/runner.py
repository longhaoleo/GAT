"""
实验流程编排层。

这个文件只负责把各个模块串起来，不放具体算法细节：
- data.py 负责读取数据、筛选特征、标准化和划分数据
- graph.py 负责构图和创建 PyG DataLoader
- gat.py 负责 GeneGAT 模型、预测和图特征提取
- train.py 负责 GeneGAT 训练循环
- xgboost.py 负责 raw / graph / hybrid 三类特征的 XGBoost 分类
- metrics.py 负责 Accuracy / F1 / AUC 等指标计算
- explain.py 负责 SHAP 可解释性分析

"""

import os

import numpy as np
import pandas as pd
import torch

from gnns.gat import (
    GeneGAT,
    extract_pooling_features,
    predict_with_loader,
)
from gnns.data import prepare_data
from gnns.graph import create_data_objects, create_graph, get_device
from gnns.train import train_graph_classifier
from pipline.explain import perform_shap_analysis
from pipline.xgboost import DEFAULT_XGB_PARAMS, xgboost_train_or_search


device = get_device()


def run_pipeline(args):
    """运行完整实验；如果设置 k_fold，会循环跑多折并汇总结果。"""
    # CUDA_LAUNCH_BLOCKING 让 GPU 报错更容易定位到具体代码行。
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # 所有输出目录都在入口处创建，后面各模块只负责写自己的结果。
    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Using device: {device}")

    all_rows = []
    best_name = None
    for fold_idx in range(args.k_fold):
        print(f"\n{'=' * 20} 第 {fold_idx + 1}/{args.k_fold} 折 {'=' * 20}")
        rows, best_name = run_fold(args, fold_idx)
        all_rows.extend(rows)

    write_summary(all_rows, args.results_dir, best_name)


def run_fold(args, fold_idx):
    """运行单折实验：数据 -> GAT -> 图嵌入 -> XGBoost -> 可选 SHAP。"""
    # k_fold=1 时使用普通 train/val/test 划分；k_fold>1 时使用当前 fold。
    fold_arg = fold_idx if args.k_fold > 1 else None
    data = prepare_data(fold_idx=fold_arg, k_fold=args.k_fold)

    # prepare_data 返回了全量张量、分组数据、基因名和 mask。
    # 当前主流程只需要 train/val/test 和 gene_names。
    _, _, X_train, y_train, X_val, y_val, X_test, y_test, gene_names, *_ = data

    # 每个样本会被转换成一张 PyG 图：
    # 节点是基因，节点特征是该样本的基因表达值。
    # 所有样本共用同一个 edge_index，也就是同一套基因关系图。
    loaders = build_loaders(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        graph_method=args.graph_method,
        batch_size=args.batch_size,
    )

    # 训练或加载 GeneGAT；threshold 是二分类概率阈值，当前默认 0.5。
    model, threshold = load_or_train_gnn(args, loaders, fold_idx)

    # 先单独评估 GAT 本身，便于和后面的 XGBoost 融合方案比较。
    _, _, _, gnn_metrics = predict_with_loader(model, loaders["test"], device, threshold=threshold)

    # 从训练好的 GAT 中提取图级 embedding。
    # 这些 embedding 代表模型学到的“基因关系模式”。
    graph_features, graph_labels = extract_graph_features(model, loaders)

    # 构造三组消融特征：
    # raw: 原始基因表达；graph: GAT 图嵌入；hybrid: 二者拼接。
    raw_data = (X_train, y_train, X_val, y_val, X_test, y_test, gene_names)
    feature_sets = make_feature_sets(raw_data, graph_features, graph_labels)
    xgb_results, best_name = train_feature_models(feature_sets)

    # SHAP 图比较耗时，且多折时每折都画会很乱；默认只画第 0 折。
    if not args.skip_shap and (args.k_fold == 1 or fold_idx == 0):
        run_hybrid_shap(args, xgb_results, fold_idx)

    return collect_rows(fold_idx, gnn_metrics, xgb_results), best_name


def build_loaders(X_train, y_train, X_val, y_val, X_test, y_test, graph_method, batch_size):
    """根据训练集构图，然后给 train/val/test 复用同一套基因图结构。"""
    # 注意：这里的图结构由训练集估计，避免使用测试集信息构图造成泄漏。
    edge_index = create_graph(X_train, method=graph_method, device=device)
    return {
        "train": create_data_objects(X_train, y_train, edge_index, batch_size=batch_size, device=device),
        "val": create_data_objects(X_val, y_val, edge_index, batch_size=batch_size, device=device),
        "test": create_data_objects(X_test, y_test, edge_index, batch_size=batch_size, device=device),
    }


def load_or_train_gnn(args, loaders, fold_idx):
    """如果给了模型路径就加载，否则训练 GeneGAT 并保存权重。"""
    model = build_model(args)

    if args.load_model:
        checkpoint = torch.load(args.load_model, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        threshold = checkpoint.get("threshold", 0.5)
        print(f"模型已加载: {args.load_model}")
        return model, threshold

    # class_weights=[1, 3] 表示晚期类别权重更高，用来缓解类别不均衡。
    model, threshold = train_graph_classifier(
        model,
        loaders["train"],
        loaders["val"],
        epochs=args.epochs,
        patience=args.patience,
        class_weights=[1, 3],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model_path = os.path.join(args.model_dir, f"model_fold_{fold_idx}.pt")
    torch.save({"model_state_dict": model.state_dict(), "threshold": threshold}, model_path)
    print(f"模型已保存: {model_path}")
    return model, threshold


def build_model(args):
    """按命令行参数创建 GeneGAT。这里保持模型结构集中，便于调参。"""
    return GeneGAT(
        input_dim=1,
        output_dim=2,
        hidden_dim=args.hidden_dim,
        pooling_dim=args.pooling_dim,
        num_heads=args.num_heads,
        gat_layers=args.gat_layers,
        dropout=args.dropout,
        activation=args.activation,
        cluster_num=args.cluster_num,
    ).to(device)


def extract_graph_features(model, loaders):
    """分别提取 train/val/test 的 GAT 池化特征。"""
    features = {}
    labels = {}
    for split, loader in loaders.items():
        split_features, split_labels = extract_pooling_features(model, loader, device)
        features[split] = split_features
        labels[split] = split_labels
    return features, labels


def make_feature_sets(raw_data, graph_features, graph_labels):
    """准备 XGBoost 的三组输入，用于做消融实验。"""
    X_train, y_train, X_val, y_val, X_test, y_test, gene_names = raw_data

    # 图特征没有生物学名字，用 Emb_i 标记；原始特征沿用基因名。
    emb_names = [f"Emb_{idx}" for idx in range(graph_features["train"].shape[1])]
    raw_names = gene_names if gene_names is not None else [f"Gene_{idx}" for idx in range(X_train.shape[1])]

    return {
        "raw": {
            "train": as_numpy(X_train),
            "val": as_numpy(X_val),
            "test": as_numpy(X_test),
            "y_train": as_numpy(y_train),
            "y_val": as_numpy(y_val),
            "y_test": as_numpy(y_test),
            "feature_names": raw_names,
        },
        "graph": {
            "train": as_numpy(graph_features["train"]),
            "val": as_numpy(graph_features["val"]),
            "test": as_numpy(graph_features["test"]),
            "y_train": as_numpy(graph_labels["train"]),
            "y_val": as_numpy(graph_labels["val"]),
            "y_test": as_numpy(graph_labels["test"]),
            "feature_names": emb_names,
        },
        "hybrid": {
            # hybrid 是这个项目最重要的对比：保留基因可解释性，同时加入图关系信息。
            "train": np.hstack([as_numpy(X_train), as_numpy(graph_features["train"])]),
            "val": np.hstack([as_numpy(X_val), as_numpy(graph_features["val"])]),
            "test": np.hstack([as_numpy(X_test), as_numpy(graph_features["test"])]),
            "y_train": as_numpy(y_train),
            "y_val": as_numpy(y_val),
            "y_test": as_numpy(y_test),
            "feature_names": raw_names + emb_names,
        },
    }


def train_feature_models(feature_sets):
    """对 raw / graph / hybrid 三组特征分别训练 XGBoost，并记录最佳结果。"""
    results = {}
    best_name = None
    best_accuracy = -1

    for name, dataset in feature_sets.items():
        print(f"\n使用 {name} 特征训练XGBoost...")
        model, params, metrics = xgboost_train_or_search(
            dataset["train"],
            dataset["y_train"],
            dataset["val"],
            dataset["y_val"],
            dataset["test"],
            dataset["y_test"],
            user_params=DEFAULT_XGB_PARAMS,
            param_dist=None,
            n_iter=30,
            random_state=42,
        )
        results[name] = {
            "model": model,
            "params": params,
            "metrics": metrics,
            "feature_names": dataset["feature_names"],
            "test_features": dataset["test"],
        }

        if metrics["accuracy"] > best_accuracy:
            best_name = name
            best_accuracy = metrics["accuracy"]

    return results, best_name


def run_hybrid_shap(args, xgb_results, fold_idx):
    """只对 hybrid 模型做 SHAP，因为它同时包含原始基因和图嵌入。"""
    hybrid = xgb_results.get("hybrid")
    if hybrid is None:
        return

    output_dir = os.path.join(args.image_dir, f"fold_{fold_idx}_hybrid_shap")
    _shap_values, feature_importance = perform_shap_analysis(
        hybrid["model"],
        hybrid["test_features"],
        feature_names=hybrid["feature_names"],
        output_dir=output_dir,
    )

    if feature_importance is None:
        return

    importance_df = pd.DataFrame(
        {"Feature": hybrid["feature_names"], "Importance": feature_importance}
    ).sort_values("Importance", ascending=False)
    importance_df.to_csv(os.path.join(output_dir, "overall_feature_importance.csv"), index=False)
    print(f"Hybrid SHAP 已保存: {output_dir}")


def collect_rows(fold_idx, gnn_metrics, xgb_results):
    """把 GAT 和三组 XGBoost 的指标整理成表格行。"""
    rows = [
        {
            "fold": fold_idx,
            "model": "gnn",
            "accuracy": gnn_metrics.get("accuracy"),
            "f1_score": gnn_metrics.get("f1_score"),
            "auc": gnn_metrics.get("auc"),
        }
    ]

    for name, result in xgb_results.items():
        metrics = result["metrics"]
        rows.append(
            {
                "fold": fold_idx,
                "model": f"xgboost_{name}",
                "accuracy": metrics.get("accuracy"),
                "f1_score": metrics.get("f1_score"),
                "auc": metrics.get("auc"),
            }
        )

    return rows


def write_summary(rows, results_dir, best_name):
    """把所有折的指标保存成 CSV 和 Markdown，方便写报告或面试复盘。"""
    summary_df = pd.DataFrame(rows)

    csv_path = os.path.join(results_dir, "metrics_summary.csv")
    md_path = os.path.join(results_dir, "metrics_summary.md")
    summary_df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as file:
        file.write("# Experiment Summary\n\n")
        file.write(format_markdown_table(summary_df))
        file.write("\n\n")
        file.write(f"Best XGBoost feature set in the last fold: `{best_name}`\n")

    print(f"\n结果汇总已保存: {csv_path}")
    print(f"结果汇总已保存: {md_path}")


def format_markdown_table(df):
    """避免额外依赖 tabulate，手写一个简单 Markdown 表格。"""
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def as_numpy(value):
    """XGBoost / sklearn 更习惯 NumPy，这里统一做 Tensor 到 NumPy 的转换。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
