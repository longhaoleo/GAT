import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class DataProcessor:
    """
    数据处理器，用于数据预处理、特征提取和数据增强
    """
    def __init__(self, device=None):
        """
        初始化数据处理器
        
        参数:
            device: 计算设备
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"数据处理器使用设备: {self.device}")
    
    def preprocess_data(self, X, y, pca_components=0.95, feature_selection=False,apply_pca=False):
        """
        预处理数据，包括PCA降维和特征选择
        
        参数:
            X: 输入特征
            y: 标签
            pca_components: PCA组件数量或方差保留比例
            feature_selection: 是否进行特征选择
            
        返回:
            处理后的特征和标签
        """
        # 确保输入是numpy数组
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X
            
        if isinstance(y, torch.Tensor):
            y_np = y.numpy()
        else:
            y_np = y
        if apply_pca:
            # PCA降维
            pca = PCA(n_components=pca_components)
            X_np = pca.fit_transform(X_np)
            print(f"PCA降维后特征维度: {X_np.shape[1]}, 保留了{pca.explained_variance_ratio_.sum()*100:.2f}%的方差")
            
        # 特征选择
        if feature_selection:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf.fit(X_np, y_np)
            
            selector = SelectFromModel(rf, threshold='median')
            X_selected = selector.fit_transform(X_np, y_np)
            print(f"特征选择后的维度: {X_selected.shape[1]}")
            
            # 转换为PyTorch张量
            X_processed = torch.from_numpy(X_selected).float()
        else:
            # 转换为PyTorch张量
            X_processed = torch.from_numpy(X_np).float()
        
        y_processed = torch.from_numpy(y_np).long()
        
        return X_processed, y_processed
    
    def augment_minority_samples(self, X, y, minority_class=1, num_augmentations=10, noise_level=0.15, mixup=True):
        """
        对少数类进行数据增强
        
        参数:
            X: 输入特征
            y: 标签
            minority_class: 少数类标签
            num_augmentations: 增强次数
            noise_level: 噪声水平
            mixup: 是否使用mixup增强
            
        返回:
            增强后的特征和标签
        """
        # 确保输入是numpy数组
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X
            
        if isinstance(y, torch.Tensor):
            y_np = y.numpy()
        else:
            y_np = y
        
        # 找出少数类样本
        minority_indices = np.where(y_np == minority_class)[0]
        majority_indices = np.where(y_np != minority_class)[0]
        X_minority = X_np[minority_indices]
        
        augmented_samples = []
        augmented_labels = []
        
        # 添加随机噪声增强
        for i in range(num_augmentations):
            noise = np.random.normal(0, noise_level * (i/num_augmentations + 0.5), X_minority.shape)
            augmented = X_minority + noise
            augmented_samples.append(augmented)
            augmented_labels.append(np.ones(len(X_minority)) * minority_class)
        
        # 如果启用mixup，添加mixup增强
        if mixup and len(minority_indices) >= 2:
            # 增加mixup样本数量
            for _ in range(num_augmentations):
                idx1, idx2 = np.random.choice(len(X_minority), 2, replace=True)
                alpha = np.random.beta(0.2, 0.2)  # 更极端的Beta分布参数
                mixed = alpha * X_minority[idx1] + (1 - alpha) * X_minority[idx2]
                augmented_samples.append(mixed.reshape(1, -1))
                augmented_labels.append(np.array([minority_class]))
                
            # 添加少数类与多数类的mixup (边界增强)
            for _ in range(num_augmentations // 2):
                idx_min = np.random.choice(len(X_minority))
                idx_maj = np.random.choice(len(majority_indices))
                # 偏向少数类的mixup (0.7-0.9)
                alpha = 0.7 + 0.2 * np.random.random()
                mixed = alpha * X_minority[idx_min] + (1 - alpha) * X_np[majority_indices[idx_maj]]
                augmented_samples.append(mixed.reshape(1, -1))
                augmented_labels.append(np.array([minority_class]))
        
        # 合并所有增强样本
        if augmented_samples:
            all_augmented_samples = np.vstack(augmented_samples)
            all_augmented_labels = np.concatenate(augmented_labels)
            
            # 合并原始数据和增强数据
            X_augmented = np.vstack([X_np, all_augmented_samples])
            y_augmented = np.concatenate([y_np, all_augmented_labels])
        else:
            X_augmented, y_augmented = X_np, y_np
        
        print(f"增强后的数据形状: X={X_augmented.shape}, y={y_augmented.shape}")
        return X_augmented, y_augmented
    
    def apply_oversampling(self, X, y, minority_class=1):
        """
        应用过采样方法
        
        参数:
            X: 输入特征
            y: 标签
            minority_class: 少数类标签
            
        返回:
            过采样后的特征和标签
        """
        # 确保输入是numpy数组
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X
            
        if isinstance(y, torch.Tensor):
            y_np = y.numpy()
        else:
            y_np = y
        
        # 先清理可能的噪声样本
        tomek = TomekLinks()
        X_cleaned, y_cleaned = tomek.fit_resample(X_np, y_np)
        print(f"Tomek Links清理后: {X_cleaned.shape}")
        
        # 使用多种过采样方法的组合
        try:
            # 先尝试BorderlineSMOTE
            k_neighbors = min(5, np.bincount(y_cleaned.astype(int))[minority_class] - 1)
            if k_neighbors < 1:
                k_neighbors = 1
            border_smote = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors)
            X_border, y_border = border_smote.fit_resample(X_cleaned, y_cleaned)
            
            # 再使用ADASYN进一步增强
            adasyn = ADASYN(random_state=42, n_neighbors=k_neighbors)
            X_resampled, y_resampled = adasyn.fit_resample(X_border, y_border)
        except ValueError as e:
            print(f"高级过采样失败: {e}")
            print("使用基本SMOTE...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_cleaned, y_cleaned)
        
        # 打印重采样后的类别分布
        unique, counts = np.unique(y_resampled, return_counts=True)
        print("重采样后的类别分布:")
        for cls, count in zip(unique, counts):
            print(f"类别 {cls}: {count} 样本")
        
        # 转换为PyTorch张量
        X_tensor = torch.from_numpy(X_resampled).float()
        y_tensor = torch.from_numpy(y_resampled).long()
        
        return X_tensor, y_tensor
    
    def prepare_data_for_training(self, X, y, test_size=0.1, val_size=0.1, 
                                apply_pca=False, apply_feature_selection=False,
                                apply_augmentation=True, apply_oversampling=True,
                                minority_class=1,
                                final_fixed_count=4000):
        """
        准备训练数据，包括预处理、数据分割、数据增强、过采样以及最后统一数据量
        
        参数:
            X: 输入特征
            y: 标签
            test_size: 测试集比例
            val_size: 验证集比例
            apply_pca: 是否应用PCA
            apply_feature_selection: 是否应用特征选择
            apply_augmentation: 是否应用数据增强
            apply_oversampling: 是否应用过采样
            minority_class: 少数类标签
            final_fixed_count: 最终训练数据固定样本数量（如果指定）
            
        返回:
            准备好的训练、验证和测试数据，以及特征维度
        """
        # 预处理数据
        if apply_pca or apply_feature_selection:
            X, y = self.preprocess_data(X, y, 
                                        pca_components=0.95, 
                                        feature_selection=apply_feature_selection,
                                        apply_pca=apply_pca)
        
        # z-score标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # 分割测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 分割验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )
        
        # 数据增强（无需固定数量参数）
        if apply_augmentation:
            X_train_aug, y_train_aug = self.augment_minority_samples(
                X_train, y_train, 
                minority_class=minority_class, 
                num_augmentations=10,
                noise_level=0.15,
                mixup=True
            )
        else:
            X_train_aug, y_train_aug = X_train, y_train
        
        # 过采样（无需固定数量参数）
        if apply_oversampling:
            X_train_final, y_train_final = self.apply_oversampling(
                X_train_aug, y_train_aug, 
                minority_class=minority_class
            )
        else:
            X_train_final, y_train_final = X_train_aug, y_train_aug

        # 最后统一训练集数据量
        if final_fixed_count is not None:
            current_count = X_train_final.shape[0]
            if current_count > final_fixed_count:
                # 数据量过多时，随机抽取固定数量样本
                indices = np.random.choice(current_count, final_fixed_count, replace=False)
                X_train_final = X_train_final[indices]
                y_train_final = y_train_final[indices]
            elif current_count < final_fixed_count:
                # 数据量不足时，重复抽样补充到固定数量
                additional_indices = np.random.choice(current_count, final_fixed_count - current_count, replace=True)
                X_additional = X_train_final[additional_indices]
                y_additional = y_train_final[additional_indices]
                X_train_final = np.vstack([X_train_final, X_additional])
                y_train_final = np.concatenate([y_train_final, y_additional])
        
        # 确保所有数据都是PyTorch张量
        if not isinstance(X_train_final, torch.Tensor):
            X_train_final = torch.from_numpy(X_train_final).float()
        if not isinstance(y_train_final, torch.Tensor):
            y_train_final = torch.from_numpy(y_train_final).long()
        
        if not isinstance(X_val, torch.Tensor):
            X_val = torch.from_numpy(X_val).float()
        if not isinstance(y_val, torch.Tensor):
            y_val = torch.from_numpy(y_val).long()
        
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.from_numpy(X_test).float()
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.from_numpy(y_test).long()
        
        # 创建PyG数据对象
        data_train = Data(x=X_train_final, y=y_train_final).to(self.device)
        data_val = Data(x=X_val, y=y_val).to(self.device)
        data_test = Data(x=X_test, y=y_test).to(self.device)
        
        return data_train, data_val, data_test, X.shape[1]
