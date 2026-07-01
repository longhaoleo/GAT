"""
特征处理和数据增强工具。

data.py 负责读取、筛选和划分数据；本文件负责可选的特征工程：
- PCA 降维
- 基于随机森林重要性的特征选择
- 少数类样本增强
- 过采样和 Tomek Links 清理
"""

import numpy as np
import torch
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


def to_numpy(value):
    """把 Tensor 或数组统一转成 NumPy，便于 sklearn / imblearn 使用。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_tensor(value, dtype=torch.float):
    """把 NumPy 数据统一转成 torch.Tensor；已是 Tensor 时只调整 dtype。"""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.from_numpy(value).to(dtype=dtype)


class DataProcessor:
    """可选特征工程模块，主要服务于旧实验和数据增强实验。"""

    def __init__(self, device=None):
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"数据处理器使用设备: {self.device}")

    def preprocess_data(self, X, y, pca_components=0.95, feature_selection=False, apply_pca=False):
        """执行 PCA 和随机森林特征选择，并返回 Tensor。

        `feature_selection=True` 时，会先训练随机森林，使用模型的
        feature_importances_ 评估每个基因的重要性；`threshold="median"`
        表示保留重要性高于中位数的基因。
        """
        X_np = to_numpy(X)
        y_np = to_numpy(y)

        if apply_pca:
            # PCA 是无监督降维：尽量保留整体方差，同时减少高维噪声。
            pca = PCA(n_components=pca_components)
            X_np = pca.fit_transform(X_np)
            kept_variance = pca.explained_variance_ratio_.sum() * 100
            print(f"PCA降维后特征维度: {X_np.shape[1]}, 保留了{kept_variance:.2f}%的方差")

        if feature_selection:
            # 随机森林能给每个基因一个 feature_importance，用来做监督式筛选。
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
            rf.fit(X_np, y_np)

            # threshold="median" 表示保留重要性高于中位数的基因。
            selector = SelectFromModel(rf, threshold="median")
            X_np = selector.fit_transform(X_np, y_np)
            print(f"特征选择后的维度: {X_np.shape[1]}")

        return to_tensor(X_np, dtype=torch.float), to_tensor(y_np, dtype=torch.long)

    def augment_minority_samples(self, X, y, minority_class=1, num_augmentations=10, noise_level=0.15, mixup=True):
        """对少数类样本做简单增强。

        增强包含两类：
        - 在少数类样本上加入高斯噪声
        - 对少数类样本做 mixup，必要时加入少数类和多数类的边界 mixup
        """
        X_np = to_numpy(X)
        y_np = to_numpy(y)

        minority_indices = np.where(y_np == minority_class)[0]
        majority_indices = np.where(y_np != minority_class)[0]
        X_minority = X_np[minority_indices]

        augmented_samples = []
        augmented_labels = []

        # 噪声增强：在少数类表达谱上加小扰动，扩充局部样本空间。
        for idx in range(num_augmentations):
            scale = noise_level * (idx / num_augmentations + 0.5)
            noise = np.random.normal(0, scale, X_minority.shape)
            augmented_samples.append(X_minority + noise)
            augmented_labels.append(np.ones(len(X_minority)) * minority_class)

        if mixup and len(minority_indices) >= 2:
            # 少数类内部 mixup：在两个少数类样本之间插值。
            for _ in range(num_augmentations):
                idx1, idx2 = np.random.choice(len(X_minority), 2, replace=True)
                alpha = np.random.beta(0.2, 0.2)
                mixed = alpha * X_minority[idx1] + (1 - alpha) * X_minority[idx2]
                augmented_samples.append(mixed.reshape(1, -1))
                augmented_labels.append(np.array([minority_class]))

            # 边界 mixup：用偏向少数类的比例混合多数类，增强决策边界附近样本。
            for _ in range(num_augmentations // 2):
                idx_min = np.random.choice(len(X_minority))
                idx_maj = np.random.choice(len(majority_indices))
                alpha = 0.7 + 0.2 * np.random.random()
                mixed = alpha * X_minority[idx_min] + (1 - alpha) * X_np[majority_indices[idx_maj]]
                augmented_samples.append(mixed.reshape(1, -1))
                augmented_labels.append(np.array([minority_class]))

        if not augmented_samples:
            return X_np, y_np

        X_augmented = np.vstack([X_np, np.vstack(augmented_samples)])
        y_augmented = np.concatenate([y_np, np.concatenate(augmented_labels)])
        print(f"增强后的数据形状: X={X_augmented.shape}, y={y_augmented.shape}")
        return X_augmented, y_augmented

    def apply_oversampling(self, X, y, minority_class=1):
        """对训练集做过采样，并返回 Tensor。

        先用 Tomek Links 清理边界噪声，再优先尝试 BorderlineSMOTE + ADASYN；
        如果数据量或类别分布不满足高级方法要求，则回退到普通 SMOTE。
        """
        X_np = to_numpy(X)
        y_np = to_numpy(y)

        # Tomek Links 先清理类别边界上的可疑样本，减少过采样时放大噪声。
        tomek = TomekLinks()
        X_cleaned, y_cleaned = tomek.fit_resample(X_np, y_np)
        print(f"Tomek Links清理后: {X_cleaned.shape}")

        try:
            # k_neighbors 不能超过少数类样本数，否则 SMOTE 系列方法会报错。
            k_neighbors = min(5, np.bincount(y_cleaned.astype(int))[minority_class] - 1)
            k_neighbors = max(k_neighbors, 1)

            # BorderlineSMOTE 优先补边界附近样本，ADASYN 再补更难分类的样本。
            border_smote = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors)
            X_border, y_border = border_smote.fit_resample(X_cleaned, y_cleaned)

            adasyn = ADASYN(random_state=42, n_neighbors=k_neighbors)
            X_resampled, y_resampled = adasyn.fit_resample(X_border, y_border)
        except ValueError as exc:
            print(f"高级过采样失败: {exc}")
            print("使用基本SMOTE...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_cleaned, y_cleaned)

        unique, counts = np.unique(y_resampled, return_counts=True)
        print("重采样后的类别分布:")
        for cls, count in zip(unique, counts):
            print(f"类别 {cls}: {count} 样本")

        return to_tensor(X_resampled, dtype=torch.float), to_tensor(y_resampled, dtype=torch.long)

    def prepare_data_for_training(
        self,
        X,
        y,
        test_size=0.1,
        val_size=0.1,
        apply_pca=False,
        apply_feature_selection=False,
        apply_augmentation=True,
        apply_oversampling=True,
        minority_class=1,
        final_fixed_count=4000,
        ):
        """旧实验入口：完成特征工程、划分、增强，并封装为 PyG Data。"""
        # 这个函数不是当前 main.py 主路径，但保留给旧的增强/过采样实验。
        if apply_pca or apply_feature_selection:
            X, y = self.preprocess_data(
                X,
                y,
                pca_components=0.95,
                feature_selection=apply_feature_selection,
                apply_pca=apply_pca,
            )

        # 标准化必须在划分前后谨慎处理；这个旧入口主要用于实验对比。
        X = StandardScaler().fit_transform(to_numpy(X))
        y = to_numpy(y)

        # 先切 test，再从训练部分切 val，避免测试集参与模型选择。
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            random_state=42,
            stratify=y_train,
        )

        if apply_augmentation:
            X_train, y_train = self.augment_minority_samples(
                X_train,
                y_train,
                minority_class=minority_class,
                num_augmentations=10,
                noise_level=0.15,
                mixup=True,
            )

        if apply_oversampling:
            X_train, y_train = self.apply_oversampling(X_train, y_train, minority_class=minority_class)

        X_train, y_train = self._fix_sample_count(X_train, y_train, final_fixed_count)

        data_train = Data(
            x=to_tensor(X_train),
            y=to_tensor(y_train, dtype=torch.long),
        ).to(self.device)
        data_val = Data(
            x=to_tensor(X_val),
            y=to_tensor(y_val, dtype=torch.long),
        ).to(self.device)
        data_test = Data(
            x=to_tensor(X_test),
            y=to_tensor(y_test, dtype=torch.long),
        ).to(self.device)
        return data_train, data_val, data_test, X.shape[1]

    def _fix_sample_count(self, X, y, final_fixed_count):
        """把训练集样本数固定到指定数量，便于复现实验规模。"""
        if final_fixed_count is None:
            return X, y

        X_np = to_numpy(X)
        y_np = to_numpy(y)
        current_count = X_np.shape[0]

        if current_count > final_fixed_count:
            indices = np.random.choice(current_count, final_fixed_count, replace=False)
            return X_np[indices], y_np[indices]

        if current_count < final_fixed_count:
            extra = np.random.choice(current_count, final_fixed_count - current_count, replace=True)
            X_np = np.vstack([X_np, X_np[extra]])
            y_np = np.concatenate([y_np, y_np[extra]])

        return X_np, y_np
