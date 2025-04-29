import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report 
import os
import datetime

def score(y_true, y_pred):
    """
    生成分类器的准确率和结构化分类报告。

    参数:
        y_true: array-like，真实标签
        y_pred: array-like，预测标签

    返回:
        accuracy: float，准确率
        report_df: pd.DataFrame，分类报告（结构化）
    """
    accuracy = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    return accuracy, report_df

def check_gene_variances(X, gene_names):
    """
    用于判断select_data_variance的阈值。
    给定特征矩阵 X (行=样本, 列=基因) 和 gene_names（列名），
    返回一个 DataFrame，包含各基因的方差，按从大到小排序。
    """
    # axis=0: 对列求方差，每列对应一个基因
    # ddof=1 是无偏估计；如果你想用总体方差，可改成 ddof=0
    variances = np.var(X, axis=0, ddof=0)

    df_var = pd.DataFrame({
        'Gene': gene_names,
        'Variance': variances
    })

    # 按方差从大到小排
    df_var.sort_values(by='Variance', ascending=False, inplace=True)
    df_var.reset_index(drop=True, inplace=True)

    return df_var
