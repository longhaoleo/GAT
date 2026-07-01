"""
SHAP 可解释性模块。

这里解释的是最终的 XGBoost hybrid 模型：
- 原始基因特征能告诉我们哪些具体基因重要
- GAT 图嵌入特征能反映模型学到的基因关系模式是否有贡献

输出的图片和 CSV 可以直接用于项目展示或面试讲解。
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def perform_shap_analysis(model, X, feature_names=None, output_dir="images/shap"):
    """计算 SHAP 值，并保存摘要图、依赖图和重要性表格。"""
    os.makedirs(output_dir, exist_ok=True)
    print("\n执行SHAP分析...")

    # 训练流程里有些数据是 Tensor；SHAP/XGBoost 使用 NumPy 更稳。
    if hasattr(X, "cpu"):
        X = X.cpu().numpy()

    # 目前只对 XGBoost 做 SHAP；其他模型需要换解释器。
    if not isinstance(model, xgb.XGBClassifier):
        print("当前模型类型不支持SHAP分析")
        return None, None

    try:
        # TreeExplainer 是树模型常用解释器，速度和稳定性都比较合适。
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # 某些 SHAP 版本二分类会返回 [class0, class1]，这里取正类。
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        save_summary_plots(shap_values, X, feature_names, output_dir)
        save_dependence_plots(shap_values, X, feature_names, output_dir)
        save_importance_tables(shap_values, feature_names, output_dir)

        mean_abs_shap = np.abs(shap_values).mean(0)
        print(f"SHAP分析完成，结果保存在 {output_dir} 目录")
        return shap_values, mean_abs_shap
    except Exception as e:
        print(f"SHAP分析失败: {str(e)}")
        return None, None


def save_summary_plots(shap_values, X, feature_names, output_dir):
    """保存两张总览图：点图看方向和分布，条形图看平均重要性。"""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_dependence_plots(shap_values, X, feature_names, output_dir):
    """对最重要的前 5 个特征画依赖图，观察特征值和贡献之间的关系。"""
    mean_abs_shap = np.abs(shap_values).mean(0)
    top_indices = np.argsort(mean_abs_shap)[-5:]

    for idx in top_indices:
        feature_name = feature_names[idx] if feature_names is not None else f"Feature {idx}"
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(idx, shap_values, X, feature_names=feature_names, show=False)
        plt.title(f"SHAP Dependence Plot - {feature_name}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"shap_dependence_{feature_name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def save_importance_tables(shap_values, feature_names, output_dir):
    """保存原始 SHAP 值和按平均绝对 SHAP 排序的重要性表。"""
    shap_df = pd.DataFrame(shap_values, columns=feature_names if feature_names is not None else None)
    shap_df.to_csv(os.path.join(output_dir, "shap_values.csv"), index=False)

    mean_abs_shap = np.abs(shap_values).mean(0)
    names = feature_names if feature_names is not None else [f"Feature_{i}" for i in range(len(mean_abs_shap))]
    importance_df = pd.DataFrame({"Feature": names, "Importance": mean_abs_shap})
    importance_df = importance_df.sort_values("Importance", ascending=False)
    importance_df.to_csv(os.path.join(output_dir, "full_feature_importance.csv"), index=False)
