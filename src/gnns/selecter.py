import numpy as np
import pandas as pd
import os
import time
from sklearn.feature_selection import VarianceThreshold
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.vectors import FactorVector
import rpy2.robjects.packages as rpackages
import rpy2.robjects.conversion as conversion
import rpy2.robjects as ro
from sklearn.preprocessing import StandardScaler

# 启用 R 与 Python 的自动转换
pandas2ri.activate()
numpy2ri.activate()

def variance(X, y, gene_names, sample_ids, threshold=0.1, **kwargs):
    """
    根据方差阈值对基因进行过滤，只保留方差大于 threshold 的基因。

    参数:
    - X: (numpy.ndarray) 样本 x 基因 (行: 样本, 列: 基因)
    - y: (numpy.ndarray) 样本对应标签
    - gene_names: (list) 基因名称，与 X 的列对应
    - sample_ids: (list) 样本ID，与 X 的行对应
    - threshold: (float) 方差阈值，默认 1
    - **kwargs: 额外参数，便于扩展

    返回:
    - X_selected: (numpy.ndarray) 过滤后保留基因的特征矩阵
    - y: 原标签不变
    - selected_gene_names: (list) 保留的基因名称
    - sample_ids: 原样本ID不变
    """
    # 参数验证
    assert X.shape[1] == len(gene_names), f"基因数不匹配: X有{X.shape[1]}列，但gene_names有{len(gene_names)}个元素"
    assert X.shape[0] == len(sample_ids), f"样本数不匹配: X有{X.shape[0]}行，但sample_ids有{len(sample_ids)}个元素"
    
    start_time = time.time()
    X = np.log1p(X)
    # 使用 sklearn 的 VarianceThreshold 进行过滤
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    
    # 获取被保留下来的列(基因)位置 True/False
    mask = selector.get_support()
    # 对应挑选后的基因名称
    selected_gene_names = [g for g, keep in zip(gene_names, mask) if keep]
    
    elapsed = time.time() - start_time
    # 方差过滤完成

    return X_selected, y, selected_gene_names, sample_ids

def process_clinical_data(clinical_file='data/TCGA_BRCA_clinical_info.csv', 
                        expression_file='data/TCGA_BRCA_expression_matrix.csv'):
    """
    读取临床信息和表达矩阵，提取样本×基因表达矩阵与标签。
    
    参数:
    - clinical_file: (str) 临床数据文件路径
    - expression_file: (str) 表达矩阵文件路径
    
    返回:
    - X: (numpy.ndarray) 样本×基因表达矩阵
    - group_labels: (numpy.ndarray) 分组标签
    - gene_names: (list) 基因名称列表
    - sample_ids: (list) 样本ID列表
    """
    # 读取临床数据和表达矩阵
    if not os.path.exists(clinical_file):
        raise FileNotFoundError(f"临床数据文件不存在: {clinical_file}")
    if not os.path.exists(expression_file):
        raise FileNotFoundError(f"表达矩阵文件不存在: {expression_file}")
    
    clinical = pd.read_csv(clinical_file)
    expr = pd.read_csv(expression_file, index_col=0)
    
    # 加载 R 包
    r('library(stats)')
    # 将临床数据中的分组直接赋值为因子，并 relevel
    r.assign('definition', FactorVector(clinical["definition"]))
    r('definition <- relevel(definition, ref="Solid Tissue Normal")')
    # 提取分组标签以便在 Python 侧查看
    group_labels = np.array(r('as.character(definition)'))
    
    # 转换表达矩阵为样本×基因
    X = expr.T.values
    gene_names = expr.index.tolist()
    sample_ids = expr.columns.tolist()
    
    return X, group_labels, gene_names, sample_ids

def DEGs1(num=10, alpha=0.05, save_path="data"):
    """
    使用 voom + limma 进行差异分析，筛选上调与下调表达基因，并保存至文件。
    仅保留 p 值显著的基因，按 logFC 排序提取上下调各 num 个基因。
    """
    pandas2ri.activate()

    # 加载数据
    X, group_labels, gene_ids, sample_ids = process_clinical_data()  # X: samples × genes
    df = pd.DataFrame(X, columns=gene_ids, index=sample_ids)
    df_R = df.T  # 转为 genes × samples，符合 R 的习惯

    r('library(edgeR)')
    r('library(limma)')

    # 传入 R
    r.assign('X_R', pandas2ri.py2rpy(df_R))
    # 构建设计矩阵直接使用 relevel 后的因子 definition
    r('design <- model.matrix(~ definition)')
    
    # 后续差异分析流程保持不变
    r('dge <- DGEList(counts = X_R)')
    r('dge <- calcNormFactors(dge)')
    r('keep <- filterByExpr(dge, design)')
    r('dge <- dge[keep,, keep.lib.sizes=FALSE]')
    r('v <- voom(dge, design)')
    r('fit <- lmFit(v, design)')
    r('fit <- eBayes(fit)')
    r('top <- topTable(fit, coef=2, number=nrow(X_R))')

    # R 转换为 pandas 数据框
    top = r['top']
    df_top = pandas2ri.rpy2py(top)
    df_top['Gene'] = list(top.rownames)

    # 筛选显著基因（adj.P.Val < alpha 且 |logFC| > 1），分别选取上调和下调各 num 个
    filt = df_top[(df_top['adj.P.Val'] < alpha) & (df_top['logFC'].abs() > 1)]
    up = filt.sort_values('logFC', ascending=False).head(num)
    down = filt.sort_values('logFC').head(num)
    selected_genes = pd.concat([up, down])['Gene'].tolist()

    print(f"[DEGs] 显著差异基因数：{len(filt)}，最终选中：{len(selected_genes)} 个")

    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, 'top_degs.txt')
    with open(out_file, 'w') as f:
        f.writelines(g + '\n' for g in selected_genes)

    print(f"[DEGs] Top genes 已保存至：{out_file}")

def DEGs0(X, y, gene_names, sample_ids, logFC_threshold=0.1, alpha=0.1, get_data=False):
    """
    基于 R limma 包实现差异表达分析，并从中筛选出 top 10 上调和 top 10 下调的基因。
    
    参数：
    -----------
    X : numpy.ndarray
        基因表达矩阵，形状为 (n_samples, n_genes)
    y : numpy.ndarray
        分组标签，形状为 (n_samples,)，例如 0 表示正常或早期，1 表示肿瘤或晚期
    gene_names : list
        基因名列表，与 X 的列一一对应
    sample_ids : list
        样本ID列表，与 X 的行一一对应
    logFC_threshold : float
        筛选基因的对数倍数阈值，默认 0.1
    alpha : float
        筛选时校正后的 p 值阈值，默认 0.1
    
    返回：
    -----------
    X_selected : numpy.ndarray
        筛选出的 top DEGs 在原始表达矩阵中的数据（样本×基因）
    y : numpy.ndarray
        原始分组标签
    selected_gene_names : list
        筛选出的 top DEGs 的基因名称列表
    sample_ids : list
        原始样本ID列表
    """

    # 激活 pandas 与 numpy 到 R 的转换
    pandas2ri.activate()
    numpy2ri.activate()

    # 1. 将 X 转换为 DataFrame
    df = pd.DataFrame(X, columns=gene_names, index=sample_ids)
    
    # 2. 转置 DataFrame供 limma 使用（limma 要求行：基因，列：样本）
    df_R = df.T  # 行为基因，列为样本
    
    # 将数据传入 R 环境
    ro.globalenv['X_R'] = pandas2ri.py2rpy(df_R)
    
    # 将 y 转换为因子（R 的 factor），并传入 R 环境
    y_factor = FactorVector([str(v) for v in y])
    ro.globalenv['y'] = y_factor

    # 3. 在 R 中构建设计矩阵：model.matrix(~ y)
    r = ro.r
    design = r('model.matrix(~ y)')
    ro.globalenv['design'] = design
    
    # 4. 加载 limma 包并进行差异表达分析
    limma = rpackages.importr("limma")
    fit = limma.lmFit(ro.globalenv['X_R'], ro.globalenv['design'])
    fit = limma.eBayes(fit)
    
    # 5. 调用 topTable 获取所有基因的统计结果
    n_genes = ro.r['nrow'](ro.globalenv['X_R'])
    top_table = limma.topTable(fit, coef=2, number=n_genes)
    
    # 将 topTable 结果转换为 pandas DataFrame
    df_top = conversion.rpy2py(top_table)

    # 取出基因名称（limma 的结果中基因名称通常存储在行名）
    gene_names_limma = [str(x) for x in list(top_table.rownames)]
    df_top['Gene'] = gene_names_limma
    
    # 6. 根据 |logFC| > logFC_threshold 和 adj.P.Val < alpha 进行初步过滤
    filtered_df = df_top[(df_top['logFC'].abs() > logFC_threshold) & (df_top['adj.P.Val'] < alpha)].reset_index(drop=True)
    
    # 7. 分别对上调和下调的基因排序，并各取 top 10
    upregulated_df = filtered_df[filtered_df['logFC'] > 0].sort_values(by='logFC', ascending=False).head(10)
    downregulated_df = filtered_df[filtered_df['logFC'] < 0].sort_values(by='logFC', ascending=True).head(10)
    
    top_DEGs = pd.concat([upregulated_df, downregulated_df], axis=0)
    selected_gene_names = top_DEGs['Gene'].tolist()
    
    # 8. 从原始 DataFrame 中提取筛选出的基因（列），得到最终表达矩阵
    df_selected = df[selected_gene_names]
    X_selected = df_selected.values

    print(f"筛选出的 DEGs 数量: {len(filtered_df)}")
    print("选出的基因名称：", selected_gene_names)
    print("筛选后 X 的形状:", X_selected.shape)
    
    if get_data:
        return df_top
    else:
        return X_selected, y, selected_gene_names, sample_ids

### 以下为模仿论文的方法对基因进行筛选

def DEGs(X, y, gene_names, sample_ids, num=30, alpha=0.05, save_path="data"):
    """
    使用 limma 进行差异分析，筛选上调与下调表达基因，并保存至文件。
    仅保留 p 值显著的基因 (adj.P.Val < alpha, 且 |logFC| > 1)，
    再分别按 logFC 排序选取上下调各 num 个基因保存。
    
    参数
    X : numpy.ndarray
        基因表达矩阵，形状为 (n_samples, n_genes)，行对应样本，列对应基因。
    y : array-like
        分组标签（例如 0 表示对照或早期，1 表示病患或晚期），形状为 (n_samples,)。
    gene_names : list
        基因名称列表，与 X 的列一一对应。
    sample_ids : list
        样本 ID 列表，与 X 的行一一对应。
    num : int, optional
        每个方向（上调或下调）选取的基因个数，默认 30。
    alpha : float, optional
        显著性阈值 (针对 adj.P.Val)，默认 0.05。
    save_path : str, optional
        保存结果文件的路径，默认 "data"。

    返回
    无显式返回值；会在指定目录下保存一个 'top_degs.txt' 文件，
    内含选出的显著上/下调基因。
    """

    # 1. 构建 pandas DataFrame, 并转置（limma 通常是行：基因 × 列：样本）
    df = pd.DataFrame(X, columns=gene_names, index=sample_ids)
    df_R = df.T  # 转为 基因 × 样本

    # 2. 在 R 中加载 limma
    r('library(limma)')

    # 3. 将表达矩阵和分组信息传给 R
    r.assign('X_R', pandas2ri.py2rpy(df_R))
    # 将 Python 端的分组标签 y 转为 R 因子
    factor_y = pd.Series(y, index=sample_ids).astype(str).astype('category')
    r.assign('y_R', pandas2ri.py2rpy(factor_y))

    # 4. 构建设计矩阵: ~ y_R（截距+分组）
    r('design <- model.matrix(~ y_R)')

    # 5. 用 limma 进行差异分析
    r('fit <- lmFit(X_R, design)')
    r('fit <- eBayes(fit)')
    r('top <- topTable(fit, coef=2, number=nrow(X_R))')

    # 6. 将 R 中差异分析结果转换回 pandas DataFrame
    top = r['top']
    df_top = pandas2ri.rpy2py(top)
    df_top['Gene'] = list(top.rownames)

    # 7. 按阈值过滤: adj.P.Val < alpha & |logFC| > 1
    filt = df_top[(df_top['adj.P.Val'] < alpha) & (df_top['logFC'].abs() > 1)]

    # 8. 分上调和下调, 各取前 num 个
    up = filt.sort_values('logFC', ascending=False).head(num)
    down = filt.sort_values('logFC', ascending=True).head(num)
    selected_genes = pd.concat([up, down])['Gene'].tolist()

    print(f"[DEGs] 显著差异基因数: {len(filt)}, 最终选中: {len(selected_genes)} 个")

    # 9. 将选出的基因写入文件
    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, 'top_degs.txt')
    with open(out_file, 'w') as f:
        for g in selected_genes:
            f.write(g + '\n')
    print(f"[DEGs] Top genes 已保存至: {out_file}")
    
###

def lxs(X, y, gene_names, sample_ids,
               low_expr_cutoff=0.8,
               sample_percent=0.2,
               var_cutoff=0.1,
               use_deg=False):
    """
    纯 Python 实现基因表达矩阵的两步过滤：
    1) 低表达过滤：在至少 sample_percent × n_samples 个样本中表达量 >= low_expr_cutoff
    2) 变异度过滤：在通过低表达过滤的基因中，保留方差 >= var_cutoff 的基因

    参数:
    -------
    X : numpy.ndarray 或 pandas.DataFrame
        基因表达矩阵，形状为 (n_samples, n_genes)
    y : array-like
        对应的标签或分组信息，可用于后续分析
    gene_names : list 或 array-like
        基因名称，长度应与 X 的列数匹配
    sample_ids : list 或 array-like
        样本 ID，长度应与 X 的行数匹配
    low_expr_cutoff : float, default=0.8
        低表达过滤的阈值
    sample_percent : float, default=0.2
        基因在多少比例的样本中需超过 low_expr_cutoff 才不被过滤
    var_cutoff : float, default=0.01
        变异度过滤的阈值
    use_deg : bool, default=False
        如果为 True，则传递到后续的差异表达分析；否则返回过滤后的结果

    返回:
    -------
    如果 use_deg 为 False:
        返回 (X_filtered, y, filtered_gene_names, sample_ids)
        X_filtered: 过滤后的表达矩阵 (numpy.ndarray)
        filtered_gene_names: 保留的基因名称 (Index 或 list)
    如果 use_deg 为 True:
        在此示例中，直接返回过滤后的数据，可扩展为调用差异表达分析函数
    """
    start_time = time.time()
    # 保证输入为 DataFrame
    if not isinstance(X, pd.DataFrame):
        df = pd.DataFrame(X, columns=gene_names, index=sample_ids)
    else:
        df = X.copy()
        df.index = sample_ids
        df.columns = gene_names

    # 如果需要，进行log(x+1)变换
    df = np.log1p(df)

    n_samples = df.shape[0]
    min_samples = sample_percent * n_samples

    # 1) 低表达过滤：统计每个基因有多少样本表达值 >= low_expr_cutoff
    counts_high = (df >= low_expr_cutoff).sum(axis=0)
    genes_after_expr = counts_high[counts_high >= min_samples].index
    df_filtered_low = df[genes_after_expr]

    # 2) 变异度过滤：计算每个基因的样本方差（ddof=1）
    variances = df_filtered_low.var(ddof=1)
    final_genes = variances[variances >= var_cutoff].index
    df_final = df_filtered_low[final_genes]

    elapsed = time.time() - start_time
    print(f"过滤完成，耗时{elapsed:.2f}秒")
    print(f"初始基因数: {df.shape[1]}")
    print(f"低表达过滤后基因数: {len(genes_after_expr)}")
    print(f"变异度过滤后基因数: {len(final_genes)}")
    print(f"过滤后数据维度: {df_final.shape}")
    print("过滤后数据预览:")
    print(df_final.head())
    
    # z-score标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(df_final)
    
    # 如果 use_deg 为 True，此处可以调用后续的差异表达分析函数 deg
    if use_deg:
        # 此处为示例，假设 deg() 已经定义
        return deg(X, y, df_final.columns, sample_ids)
    else:
        return X, y, df_final.columns, sample_ids


# 用于缓存转换后的 R 对象，避免重复转换
_limma_cache = {}

def deg(X, y, gene_names, sample_ids, 
                      logFC_threshold=0.27, 
                      alpha=0.05, 
                      sort_by='logFC',
                      keep_genes=[],
                      get_data=False,
                      save_path="data"):
    """
    使用 limma 进行差异表达分析，并利用全局缓存减少重复的数据转换开销，
    从而提升速度。
    
    参数说明与原版 deg 函数一致，这里不再赘述。
    """
    start_time = time.time()
    print(f"开始差异表达分析，基因数 = {len(gene_names)}")
    print(f"差异分析参数: logFC_threshold={logFC_threshold}, alpha={alpha}, sort_by={sort_by}")
    
    if sort_by not in ['logFC', 'pvalue']:
        raise ValueError("sort_by 参数只能是 'logFC' 或 'pvalue'")
    
    # 构造缓存 key，这里简单用 (数据维度, 分组元组) 作为 key
    key = (np.shape(X), tuple(y))
    
    # 如果缓存中不存在，则进行转换并缓存
    if key not in _limma_cache:
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X, columns=gene_names, index=sample_ids)
        else:
            df = X.copy()
            df.columns = gene_names
            df.index = sample_ids
        
        # 转置后转换为 R 的 DataFrame：行是基因、列是样本，与 limma 要求一致
        df_R = pandas2ri.py2rpy(df.T)
        ro.globalenv['X_R'] = df_R
        
        # 构建分组因子
        y_factor = FactorVector([str(v) for v in y])
        ro.globalenv['y'] = y_factor
        
        # 构建设计矩阵
        r = ro.r
        r('design <- model.matrix(~ y)')
        design = r['design']
        ro.globalenv['design'] = design
        
        # 缓存转换结果
        _limma_cache[key] = (df_R, design)
    else:
        df_R, design = _limma_cache[key]
        ro.globalenv['X_R'] = df_R
        ro.globalenv['design'] = design
        y_factor = FactorVector([str(v) for v in y])
        ro.globalenv['y'] = y_factor
    
    # 调用 limma 包进行线性模型拟合和贝叶斯修正
    r = ro.r
    try:
        # 导入 limma
        limma = ro.packages.importr("limma")
        # 拟合模型
        fit = limma.lmFit(ro.globalenv['X_R'], ro.globalenv['design'])
        fit = limma.eBayes(fit)
    
        # 获取结果表：n_genes 为总基因数
        n_genes = int(r['nrow'](ro.globalenv['X_R'])[0])
        top_table = limma.topTable(fit, coef=2, number=n_genes, sort_by="none")
        # 转换为 pandas DataFrame，并提取基因名称
        from rpy2.robjects import conversion
        df_top = conversion.rpy2py(top_table)
        df_top['Gene'] = list(top_table.rownames)
    
        if get_data:
            elapsed = time.time() - start_time
            print(f"差异分析完成，耗时 {elapsed:.2f} 秒")
            return df_top.reset_index(drop=True)
    
        # 根据阈值筛选
        filtered_df = df_top[(df_top['logFC'].abs() > logFC_threshold) &
                             (df_top['adj.P.Val'] < alpha)].copy()
    
        if filtered_df.empty:
            print("没有基因满足 logFC 和 FDR 阈值，请适当降低过滤标准。")
            return np.array([]), y, [], sample_ids
    
        # 排序
        if sort_by == 'logFC':
            filtered_df = filtered_df.sort_values('logFC', ascending=False)
        else:
            filtered_df = filtered_df.sort_values('adj.P.Val', ascending=True)
    
        # 强制保留先验基因
        if keep_genes:
            keep_genes = [
                "UCN3", "MUC2", "CGA", "CSAG1", "MAGEA12", "ACTL8", "MAGEA1",
                "IBSP", "KLHL1", "MAGEA3", "MYH2", "CKM", "MIR1–1HG", "MYL2",
                "MYH7", "PPP1R3A", "MYL1", "STRIT1", "C10orf71", "NRAP"
            ]
            print(f"额外纳入生物学先验基因: {df_keep.shape[0]} 个")
        keep_genes_set = set(keep_genes)
        df_keep = df_top[df_top['Gene'].isin(keep_genes_set)]
        # 合并过滤结果与强制保留基因（去重）
        filtered_df = pd.concat([filtered_df, df_keep]).drop_duplicates(subset=['Gene'])
        
    
        selected_gene_names = filtered_df['Gene'].tolist()
        # 原始数据（未转置）的 DataFrame
        if not isinstance(X, pd.DataFrame):
            df_orig = pd.DataFrame(X, columns=gene_names, index=sample_ids)
        else:
            df_orig = X.copy()
            df_orig.columns = gene_names
            df_orig.index = sample_ids
        df_selected = df_orig[selected_gene_names]
        X_selected = df_selected.values
    
        elapsed = time.time() - start_time
        print(f"差异分析完成，耗时 {elapsed:.2f} 秒")
        print(f"符合阈值的基因总数: {filtered_df.shape[0]}")
        print(f"最终选出的基因数量: {len(selected_gene_names)}")
        print("选出的基因名称：", selected_gene_names)
    
        # 保存结果
        os.makedirs(save_path, exist_ok=True)
        filename = f"deg_genes_fc{logFC_threshold}_p{alpha}.txt"
        out_file = os.path.join(save_path, filename)
        with open(out_file, 'w') as f:
            for gene in selected_gene_names:
                f.write(f"{gene}\n")
        print(f"[DEG] 筛选后的基因名称已保存至: {out_file}")
    
        return X_selected, np.array(y), selected_gene_names, sample_ids
    
    except Exception as e:
        print(f"差异分析过程中发生错误: {e}")
        raise