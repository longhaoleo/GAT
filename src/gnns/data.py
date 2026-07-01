"""
数据读取、基因筛选和数据划分。

本文件只处理“表达矩阵到 train / val / test 张量”的过程：
- 读取 stage I-IV 表达数据
- 映射早期 / 晚期标签
- 调用 selecter.py 做基因筛选
- 标准化表达矩阵
- 划分训练集、验证集、测试集
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import gnns.selecter as selecter
from gnns.featurer import DataProcessor


STAGE_LABELS = {
    # 二分类设定：I/II 期视为早期，III/IV 期视为晚期。
    "stagei_data.txt": 0,
    "stageii_data.txt": 0,
    "stageiii_data.txt": 1,
    "stageiv_data.txt": 1,
}


def load_data(file_path, method="None", **kwargs):
    """读取四个分期表达文件，并返回筛选后的表达矩阵。

    返回的 X 是 [样本数, 基因数]，y 是早晚期二分类标签：
    stage I/II 为 0，stage III/IV 为 1。
    """
    frames = []
    for filename, label in STAGE_LABELS.items():
        # 每个 stage 文件都是“行=样本、列=基因”；第一列作为样本 ID。
        full_path = os.path.join(file_path, filename)
        frame = pd.read_csv(full_path, sep="\t", index_col=0)

        # 显式新增标签列，后面再统一拼接四个分期的数据。
        frame["pathologic_stage"] = label
        frames.append(frame)

    df = pd.concat(frames, axis=0)
    feature_df = df.drop(columns=["pathologic_stage"])

    # X 保持 [样本数, 基因数]，这也是后续建图时的原始矩阵形状。
    X = feature_df.values
    y = df["pathologic_stage"].values
    gene_names = feature_df.columns.tolist()
    sample_ids = df.index.tolist()

    return select_data(X, y, gene_names, sample_ids, method=method, **kwargs)


def select_data(X, y, gene_names, sample_ids, method=None, read=True, use_deg=True, **kwargs):
    """根据 method 做基因筛选。

    常用入口是 `method="lxs"`：
    如果本地已有差异基因列表就直接读取；否则调用 selecter.lxs 重新分析。
    """
    if method == "variance":
        # 方差筛选：过滤掉样本间变化很小的基因。
        return selecter.variance(X, y, gene_names, sample_ids, **kwargs)

    if method == "DEGs":
        # DEGs 模式读取 top_degs.txt；没有文件时先跑差异分析生成文件。
        return _select_from_gene_file(
            X,
            y,
            gene_names,
            sample_ids,
            gene_file=os.path.join("data", "top_degs.txt"),
            fallback=lambda: selecter.DEGs(X, y, gene_names, sample_ids),
            method=method,
            **kwargs,
        )

    if method == "lxs":
        if not use_deg:
            return selecter.lxs(X, y, gene_names, sample_ids, use_deg=use_deg)

        # 当前主流程使用这个差异基因文件，避免每次运行都重复调用 R/limma。
        gene_file = os.path.join("data", "deg_genes_fc0.27_p0.05.txt")
        print(gene_file)
        if os.path.exists(gene_file) and read:
            return _select_existing_genes(X, y, gene_names, sample_ids, gene_file)

        return selecter.lxs(X, y, gene_names, sample_ids, use_deg=use_deg)

    return X, y, gene_names, sample_ids


def _select_from_gene_file(X, y, gene_names, sample_ids, gene_file, fallback, method, **kwargs):
    """读取基因列表；如果文件不存在，先运行 fallback 再重新读取。"""
    if os.path.exists(gene_file):
        return _select_existing_genes(X, y, gene_names, sample_ids, gene_file)

    fallback()
    return select_data(X, y, gene_names, sample_ids, method=method, **kwargs)


def _select_existing_genes(X, y, gene_names, sample_ids, gene_file):
    """按给定基因列表切表达矩阵，并同步更新 gene_names。"""
    with open(gene_file, "r", encoding="utf-8") as file:
        selected_genes = [line.strip() for line in file if line.strip()]

    # 差异分析文件里的基因必须和当前表达矩阵列名对齐。
    selected_genes = [gene for gene in selected_genes if gene in gene_names]
    if not selected_genes:
        raise ValueError(f"{gene_file} 存在，但没有可用于当前表达矩阵的基因。")

    # 用 DataFrame 按列名取基因，比按下标取更安全。
    df = pd.DataFrame(X, columns=gene_names, index=sample_ids)
    X_selected = df.loc[sample_ids, selected_genes].values
    return X_selected, y, selected_genes, sample_ids


def get_split_data(
    X,
    y,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    k_fold=None,
    fold_idx=None,
    augment=False,
    oversample=False,
    data_processor=None,
    minority_class=1,
    random_state=42,
):
    """旧实验兼容入口：划分数据，并可选只增强训练集。"""
    # 这个函数不是当前 main.py 的主入口，但保留给旧实验或快速测试。
    train_ratio, val_ratio, test_ratio = _normalize_ratios(train_ratio, val_ratio, test_ratio)
    train_indices, val_indices, test_indices = _split_indices(
        X.shape[0],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        k_fold=k_fold,
        fold_idx=fold_idx,
        random_state=random_state,
    )

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    if augment or oversample:
        # 增强和过采样只能作用在训练集，避免验证/测试集信息泄漏。
        data_processor = data_processor or DataProcessor()
        original_train_size = X_train.shape[0]

        if augment:
            X_train, y_train = data_processor.augment_minority_samples(
                X_train,
                y_train,
                minority_class=minority_class,
                num_augmentations=10,
                noise_level=0.15,
                mixup=True,
            )

        if oversample:
            X_train, y_train = data_processor.apply_oversampling(
                X_train,
                y_train,
                minority_class=minority_class,
            )

        print(f"数据处理后训练集大小: {X_train.shape[0]} (原始: {original_train_size})")

    X_full = np.concatenate([X_train, X_val, X_test], axis=0)
    y_full = np.concatenate([y_train, y_val, y_test], axis=0)
    train_mask, val_mask, test_mask = _build_split_masks(len(X_train), len(X_val), len(X_test))
    return X_full, y_full, train_mask, val_mask, test_mask


def prepare_data(fold_idx=None, k_fold=5):
    """主流程入口：读取数据、筛选基因、标准化并划分 train / val / test。"""
    # 这里使用 lxs 差异基因筛选，得到后续 GAT 和 XGBoost 共用的基因集合。
    X, y, gene_names, sample_ids = load_data("data", method="lxs", read=True, use_deg=True)

    # 标准化按基因列进行，让不同基因的表达尺度可比。
    X = StandardScaler().fit_transform(X)

    # GAT 使用 PyTorch；XGBoost 前会再转回 NumPy。
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()

    # k_fold=1 时使用固定单次划分；k_fold>1 时每折换一个 test fold。
    if k_fold == 1 or fold_idx is None:
        train_indices, val_indices, test_indices = split_once(len(X), random_state=42)
    else:
        train_indices, val_indices, test_indices = split_fold(X, y, fold_idx, k_fold)

    X_train = X_tensor[train_indices]
    y_train = y_tensor[train_indices]
    X_val = X_tensor[val_indices]
    y_val = y_tensor[val_indices]
    X_test = X_tensor[test_indices]
    y_test = y_tensor[test_indices]

    train_mask, val_mask, test_mask = _build_tensor_masks(
        len(X_tensor),
        train_indices,
        val_indices,
        test_indices,
    )

    print(f"数据集划分 (Fold {fold_idx if fold_idx is not None else 'Default'})：")
    print(f"  训练样本数: {len(X_train)}")
    print(f"  验证样本数: {len(X_val)}")
    print(f"  测试样本数: {len(X_test)}")

    return (
        X_tensor,
        y_tensor,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        gene_names,
        sample_ids,
        train_mask,
        val_mask,
        test_mask,
    )


def split_once(n_samples, random_state=42):
    """单次实验划分：70% train，15% val，15% test。"""
    # 先打乱样本下标，再按比例切分，保证每次随机种子一致时可复现。
    indices = np.arange(n_samples)
    np.random.seed(random_state)
    np.random.shuffle(indices)

    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.15)
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]
    return train_indices, val_indices, test_indices


def split_fold(X, y, fold_idx, k_fold):
    """K 折划分：当前 fold 做 test，其余样本再拆出 val。"""
    if fold_idx >= k_fold:
        raise ValueError(f"fold_idx 必须小于 k_fold, 当前 fold_idx={fold_idx}, k_fold={k_fold}")

    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    folds = list(skf.split(X, y))
    train_val_indices, test_indices = folds[fold_idx]

    # test fold 固定后，再从剩余 train_val 中切出一部分做验证集。
    val_size = len(train_val_indices) // 5
    np.random.seed(42 + fold_idx)
    np.random.shuffle(train_val_indices)

    train_indices = train_val_indices[val_size:]
    val_indices = train_val_indices[:val_size]
    return train_indices, val_indices, test_indices


def _normalize_ratios(train_ratio, val_ratio, test_ratio):
    """确保 train / val / test 比例和为 1。"""
    total_ratio = train_ratio + val_ratio + test_ratio
    if np.isclose(total_ratio, 1.0):
        return train_ratio, val_ratio, test_ratio

    return train_ratio / total_ratio, val_ratio / total_ratio, test_ratio / total_ratio


def _split_indices(n_samples, train_ratio, val_ratio, test_ratio, k_fold, fold_idx, random_state):
    """为旧实验接口生成 train / val / test 索引。"""
    indices = np.arange(n_samples)
    np.random.seed(random_state)
    np.random.shuffle(indices)

    if k_fold is not None and fold_idx is not None:
        if fold_idx >= k_fold:
            raise ValueError(f"fold_idx 必须小于 k_fold, 当前 fold_idx={fold_idx}, k_fold={k_fold}")

        test_size = int(n_samples * test_ratio)
        test_indices = indices[:test_size]
        train_val_indices = indices[test_size:]

        kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)
        train_val_splits = list(kf.split(train_val_indices))
        train_idx_rel, val_idx_rel = train_val_splits[fold_idx]
        return train_val_indices[train_idx_rel], train_val_indices[val_idx_rel], test_indices

    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]
    return train_indices, val_indices, test_indices


def _build_split_masks(n_train, n_val, n_test):
    """为旧实验接口生成 NumPy 布尔 mask。"""
    train_mask = np.array([True] * n_train + [False] * (n_val + n_test))
    val_mask = np.array([False] * n_train + [True] * n_val + [False] * n_test)
    test_mask = np.array([False] * (n_train + n_val) + [True] * n_test)
    return train_mask, val_mask, test_mask


def _build_tensor_masks(n_samples, train_indices, val_indices, test_indices):
    """为主流程生成 Tensor 布尔 mask。"""
    train_mask = torch.zeros(n_samples, dtype=torch.bool)
    val_mask = torch.zeros(n_samples, dtype=torch.bool)
    test_mask = torch.zeros(n_samples, dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    return train_mask, val_mask, test_mask
