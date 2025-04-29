import pandas as pd
import numpy as np
import gnns.selecter as selecter
import os
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, add_self_loops
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

def load_data(file_path, method='None', **kwargs):
    """
    从已经“行=样本、列=基因”的CSV文件中加载基因表达数据，
    并根据文件名给所有样本打上对应的分期标签。

    参数:
    - file_path: str, CSV 文件的完整路径（含文件名，比如 "stagei_data.csv"）。

    返回:
    - X: (numpy.ndarray) 样本的基因表达数据 (行: 样本, 列: 基因)
    - y: (numpy.ndarray) 样本对应的标签 (分期)
    - gene_names: (list) 基因列名称
    - sample_ids: (list) 样本ID (行索引)
    """

    # 根据文件名为不同阶段指定标签
    # file_stage_map = {
    #     "stagei_data.txt": 1,
    #     "stageii_data.txt": 2,
    #     "stageiii_data.txt": 3,
    #     "stageiv_data.txt": 4
    # }
    # 二分类使用如下代码
    file_stage_map = {
        "stagei_data.txt": 0,
        "stageii_data.txt": 0,
        "stageiii_data.txt": 1,
        "stageiv_data.txt": 1
    }
    #####

    df_list = []
    for fname, stage in file_stage_map.items():
        # 拼接成完整路径
        full_path = file_path + '/' + fname
        # 第1列就是样本ID，因此用 index_col=0 读入后，行索引是样本ID
        df_stage = pd.read_csv(full_path, sep='\t', index_col=0)
        df_stage.pathologic_stage = stage
        df_list.append(df_stage)
    
    # 拼接所有阶段的DataFrame
    df = pd.concat(df_list, axis=0)
    # 把特征矩阵 X = 除“pathologic_stage”列之外的所有列
    X = df.drop(columns=["pathologic_stage"]).values
    y = df["pathologic_stage"].values


    # 基因名称 = 除“pathologic_stage”列以外的所有列名
    gene_names = df.drop(columns=["pathologic_stage"]).columns.tolist()
    sample_ids = df.index.tolist()
    
    return select_data(X, y, gene_names, sample_ids, method=method, **kwargs)

def load_data3(file_path, method='DEGs', **kwargs):
    # 读取数据
    clinical_df = pd.read_csv(f"{file_path}/TCGA_BRCA_clinical_info.csv")
    expr_df = pd.read_csv(f"{file_path}/TCGA_BRCA_expression_matrix.csv", index_col=0).T  # 转置：行是样本ID

    # 统一小写、去空格
    clinical_df['ajcc_pathologic_stage'] = clinical_df['ajcc_pathologic_stage'].astype(str).str.lower().str.strip()

    # 映射表（早期为0，晚期为1）
    stage_label_map = {
        'stage 0': 0,
        'stage i': 0, 'stage ia': 0, 'stage ib': 0,
        'stage ii': 0, 'stage iia': 0, 'stage iib': 0,
        'stage iii': 1, 'stage iiia': 1, 'stage iiib': 1, 'stage iiic': 1,
        'stage iv': 1
        # 其他如 stage x, na 不在映射中，会被 drop 掉
    }

    # 多分类
    # stage_label_map = {
    # 'stage 0': 0,
    # 'stage i': 1, 'stage ia': 1, 'stage ib': 1,
    # 'stage ii': 2, 'stage iia': 2, 'stage iib': 2,
    # 'stage iii': 3, 'stage iiia': 3, 'stage iiib': 3, 'stage iiic': 3,
    # 'stage iv': 4
    # }

    clinical_df['label'] = clinical_df['ajcc_pathologic_stage'].map(stage_label_map)
    clinical_df.dropna(subset=['label'], inplace=True)


    # 只保留表达矩阵中存在的样本
    valid_ids = [sid for sid in clinical_df['barcode'] if sid in expr_df.index]
    clinical_df = clinical_df[clinical_df['barcode'].isin(valid_ids)]
    expr_df = expr_df.loc[valid_ids]

    # 取出筛选出的样本，及其对应的标签
    clinical_df.set_index('barcode', inplace=True)
    y = clinical_df.loc[expr_df.index]['label'].values.astype(int)

    # 特征数据准备
    X = expr_df.values
    gene_ids = expr_df.columns.tolist()
    sample_ids = expr_df.index.tolist()
    # # z-score标准化
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # 特征选择
    return select_data(X, y, gene_ids, sample_ids, method=method, **kwargs)

def select_data(X, y, gene_names, sample_ids, method=None,read=True,use_deg=True, **kwargs):
    if method == 'variance':
        X, y, gene_names, sample_ids = selecter.variance(X, y, gene_names, sample_ids, **kwargs)
    elif method == 'DEGs':
        deg_file = os.path.join('data', 'top_degs.txt')
        if os.path.exists(deg_file):
            # 读取基因列表
            with open(deg_file, 'r') as f:
                selected_genes = [line.strip() for line in f]
            # 若有需要，可以再过滤只保留实际存在于 gene_names 的基因
            selected_genes = [g for g in selected_genes if g in gene_names]
            if not selected_genes:
                raise ValueError("top_degs.txt 文件存在但内容为空，或基因均不在表达矩阵列中！")
            # 用 DataFrame 方式切分矩阵
            df = pd.DataFrame(X, columns=gene_names, index=sample_ids)
            # 这里显式按 sample_ids 排列（有些情况可能行顺序已对齐，这一步可视情况决定）
            df = df.loc[sample_ids]
            # 按选出的基因取列
            X = df.loc[:, selected_genes].values
            # 将 gene_names 更新为切后的基因名列表，保证返回值列与 gene_names 一致
            gene_names = selected_genes
        else:
            # 如果没找到 top_degs.txt，就调用 DEGs 分析，再递归一次
            selecter.DEGs(X, y, gene_names, sample_ids)
            # 分析完成后，文件已写入，再调用本函数读取即可
            return select_data(X, y, gene_names, sample_ids, method=method, **kwargs)
    elif method == 'lxs':
        if use_deg:
            name = 'deg_genes_fc0.27_p0.05.txt'
            deg_file = os.path.join('data', name)
            print(deg_file)
            if os.path.exists(deg_file) and read:
                # 读取基因列表
                with open(deg_file, 'r') as f:
                    selected_genes = [line.strip() for line in f]
                # 若有需要，可以再过滤只保留实际存在于 gene_names 的基因
                selected_genes = [g for g in selected_genes if g in gene_names]
                if not selected_genes:
                    raise ValueError("{name}文件存在但内容为空！")
                # 用 DataFrame 方式切分矩阵
                df = pd.DataFrame(X, columns=gene_names, index=sample_ids)
                # 这里显式按 sample_ids 排列（有些情况可能行顺序已对齐，这一步可视情况决定）
                df = df.loc[sample_ids]
                # 按选出的基因取列
                X = df.loc[:, selected_genes].values
                # 将 gene_names 更新为切后的基因名列表，保证返回值列与 gene_names 一致
                gene_names = selected_genes
            else:
                # 如果没找到，就调用 DEGs 分析
                return selecter.lxs(X, y, gene_names, sample_ids,use_deg=use_deg)
                
        else:
            X, y, gene_names, sample_ids = selecter.lxs(X, y, gene_names, sample_ids,use_deg=use_deg)
    else:
        # 如果没有指定特定的选择方式，则什么都不做
        pass

    return X, y, gene_names, sample_ids


def get_split_data(X, y, 
                   train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, 
                   k_fold=None, fold_idx=None, 
                   augment=False, oversample=False,
                   data_processor=None,   # DataProcessor 实例，需要包含 augment_minority_samples 和 apply_oversampling 方法
                   minority_class=1,
                   random_state=42):
    """
    划分数据为训练、验证、测试集，并可选择性对训练集进行数据增强和过采样，
    同时构造全数据集以及各个集合对应的布尔型掩码（mask）。
    
    参数:
      - X: 特征数据，numpy.ndarray (n_samples, n_features)
      - y: 标签数据，numpy.ndarray (n_samples,)
      - train_ratio: 训练集比例（默认70%）
      - val_ratio: 验证集比例（默认20%）
      - test_ratio: 测试集比例（默认10%）
      - k_fold: 如果不为 None，则使用 k 折交叉验证模式
      - fold_idx: 当前折的索引（仅当 k_fold 不为 None 时有效）
      - augment: 是否对训练集应用数据增强（增强少数类样本）
      - oversample: 是否对训练集应用过采样方法
      - data_processor: DataProcessor 实例，要求提供 augment_minority_samples 和 apply_oversampling 方法
      - minority_class: 少数类标签，默认1
      - random_state: 随机种子
    
    返回:
      - X_full: 合并后的全数据集（训练、验证、测试）(numpy.ndarray)
      - y_full: 全数据集标签 (numpy.ndarray)
      - train_mask: 布尔型掩码，指示哪些样本属于训练集 (numpy.ndarray)
      - val_mask: 布尔型掩码，指示哪些样本属于验证集 (numpy.ndarray)
      - test_mask: 布尔型掩码，指示哪些样本属于测试集 (numpy.ndarray)
    """
    import numpy as np
    from sklearn.model_selection import KFold

    # 保证比例和为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        train_ratio = train_ratio / total_ratio
        val_ratio   = val_ratio / total_ratio
        test_ratio  = test_ratio / total_ratio

    n_samples = X.shape[0]
    
    if k_fold is not None and fold_idx is not None:
        if fold_idx >= k_fold:
            raise ValueError(f"fold_idx 必须小于 k_fold, 当前 fold_idx={fold_idx}, k_fold={k_fold}")
        
        # 打乱所有样本索引
        indices = np.arange(n_samples)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        test_size = int(n_samples * test_ratio)
        test_indices = indices[:test_size]
        train_val_indices = indices[test_size:]
        
        # 对剩余数据用 k 折交叉验证划分训练集和验证集
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)
        train_val_splits = list(kf.split(train_val_indices))
        train_idx_rel, val_idx_rel = train_val_splits[fold_idx]
        
        train_indices = train_val_indices[train_idx_rel]
        val_indices   = train_val_indices[val_idx_rel]
    else:
        # 标准划分
        indices = np.arange(n_samples)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
    
    # 提取初始划分数据
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val   = X[val_indices]
    y_val   = y[val_indices]
    X_test  = X[test_indices]
    y_test  = y[test_indices]
    
    # 对训练集进行数据增强和/或过采样（仅训练集数据处理）
    if (augment or oversample):
        # 如果没有提供数据处理器，则创建一个
        if data_processor is None:
            data_processor = DataProcessor()
        
        # 保存原始训练集大小，用于后续创建掩码
        original_train_size = X_train.shape[0]
        
        if augment:
            X_train, y_train = data_processor.augment_minority_samples(
                X_train, y_train, 
                minority_class=minority_class, 
                num_augmentations=10, 
                noise_level=0.15, 
                mixup=True
            )
        if oversample:
            X_train, y_train = data_processor.apply_oversampling(
                X_train, y_train, 
                minority_class=minority_class
            )
        
        # 打印增强/过采样后的训练集大小
        print(f"数据处理后训练集大小: {X_train.shape[0]} (原始: {original_train_size})")
    
    # 合并训练、验证和测试集，构成全数据集
    X_full = np.concatenate([X_train, X_val, X_test], axis=0)
    y_full = np.concatenate([y_train, y_val, y_test], axis=0)
    
    # 根据各集合的样本数构建对应的布尔型掩码
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]
    
    train_mask = np.array([True] * n_train + [False] * (n_val + n_test))
    val_mask   = np.array([False] * n_train + [True] * n_val + [False] * n_test)
    test_mask  = np.array([False] * (n_train + n_val) + [True] * n_test)
    
    # 返回合并后的数据集和掩码
    return X_full, y_full, train_mask, val_mask, test_mask

def get_graph_data(X,y,corr_method='pearson',threshold=0.8):
    """
    根据指定方法(皮尔逊/斯皮尔曼)计算基因两两之间的相关系数(只考虑大小，取绝对值)，得到邻接矩阵。
    低于指定阈值的相关性设为 0。
    
    返回: adjacency, X, y, gene_names, sample_ids
    - adjacency: 形状为 (n_genes, n_genes) 的邻接矩阵
    - X, y, gene_names, sample_ids: 同 load_data
    
    参数:
    - file_path: str, CSV 文件路径
    - corr_method: str, 'pearson' 或 'spearman'
    - threshold: float, 低于该值的相关系数设为 0
    """
    if corr_method.lower() == 'pearson':
        adjacency = np.corrcoef(X.cpu(), rowvar=False)
    elif corr_method.lower() == 'spearman':
        corr_mat, _ = spearmanr(X.cpu(), axis=0)
        adjacency = corr_mat
    else:
        raise ValueError("corr_method must be 'pearson' or 'spearman'")
    
    # 应用阈值，低于 threshold 的值设为 0
    if threshold != 0:
        adjacency[np.abs(adjacency) <= threshold] = 0
        adjacency[np.abs(adjacency) > threshold] = 1
    return np.abs(adjacency), X ,y


def get_edge_index(X, y, corr_method='pearson', threshold=0.8, self_loops=False):
    '''
    构建基于 PyG 的 edge_index。

    参数:
        X: np.ndarray, shape = [num_samples, num_genes]
        y: np.ndarray, 标签数组
        corr_method: str, 'pearson' 或 'spearman'
        threshold: float, 相关性阈值，低于此值的边将被删除
        self_loops: bool, 是否保留自环，默认 False

    返回:
        edge_index: torch.LongTensor, 形状为 [2, num_edges]
    '''
    # 1. 计算邻接矩阵
    adjacency, _, _ = get_graph_data(X, y, corr_method=corr_method, threshold=threshold)

    # 2. 可选地去除自环（对角线）
    if not self_loops:
        np.fill_diagonal(adjacency, 0)

    # 3. 获取非零位置并转换为 edge_index
    row, col = np.nonzero(adjacency)
    edge_index = torch.tensor([row, col], dtype=torch.long)

    return edge_index


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


