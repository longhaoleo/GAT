"""
指标计算工具。

这个文件集中放模型评估相关代码，避免 GAT、XGBoost 等模块里重复写
accuracy / F1 / AUC 的计算逻辑。
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score


def to_numpy(values):
    """把 torch.Tensor 或普通数组统一转成 numpy.ndarray。"""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def evaluate_classification(y_true, y_pred, y_prob=None, average="binary", threshold=None):
    """统一计算分类指标。

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率。二分类时可以传 [n_samples, 2] 或 [n_samples]
        average: F1 的平均方式。GAT 用 binary，XGBoost 默认用 weighted
        threshold: 可选，记录当前二分类阈值
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average=average),
        "auc": None,
    }

    if threshold is not None:
        metrics["threshold"] = threshold

    if y_prob is None:
        return metrics

    y_prob = to_numpy(y_prob)
    try:
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
        elif y_prob.ndim == 1:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        # 测试集只有一个类别等情况无法计算 AUC，不中断主实验。
        metrics["auc"] = None

    return metrics


def print_metrics(prefix, metrics):
    """把指标打印成统一格式，方便扫日志。"""
    auc_text = f"{metrics['auc']:.4f}" if metrics.get("auc") is not None else "N/A"
    print(
        f"{prefix} -> Accuracy: {metrics['accuracy']:.4f}, "
        f"F1: {metrics['f1_score']:.4f}, AUC: {auc_text}"
    )


def score(y_true, y_pred):
    """兼容旧实验：返回 accuracy 和 sklearn classification_report 表。"""
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return accuracy, pd.DataFrame(report).transpose()


def check_gene_variances(X, gene_names):
    """查看每个基因的方差，常用于判断方差过滤阈值。"""
    variances = np.var(X, axis=0, ddof=0)
    df_var = pd.DataFrame({"Gene": gene_names, "Variance": variances})
    df_var.sort_values(by="Variance", ascending=False, inplace=True)
    df_var.reset_index(drop=True, inplace=True)
    return df_var
