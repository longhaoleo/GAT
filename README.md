# GeneGAT 乳腺癌分期预测

本项目使用 TCGA-BRCA / 分期基因表达数据做早晚期二分类。核心流程是：

1. 读取基因表达矩阵和临床分期标签
2. 低表达过滤、方差过滤、limma 差异表达基因筛选
3. 将每个样本构造成基因图
4. 使用 GeneGAT 提取图嵌入
5. 比较 raw、graph、hybrid 三种 XGBoost 分类结果
6. 使用 SHAP 分析重要基因和图嵌入贡献

更详细的项目速读见 `PROJECT_GUIDE.md`。

# 代码入口

- `main.py`: 推荐入口，流程更清晰，适合复习和展示。
- `src/pipline/runner.py`: 串起完整实验流程。
- `src/pipline/xgboost.py`: XGBoost 训练、参数搜索和评估。
- `src/pipline/explain.py`: SHAP 分析和解释图保存。
- `src/gnns/data.py`: 数据读取、标签映射、特征筛选、标准化和数据划分。
- `src/gnns/featurer.py`: PCA、随机森林特征选择、数据增强和过采样。
- `src/gnns/graph.py`: 底层建边函数、基因图构建和 PyG DataLoader 创建。
- `src/gnns/metrics.py`: Accuracy、F1、AUC 等统一指标计算。
- `src/gnns/selecter.py`: 基因筛选和差异表达分析。
- `src/gnns/gat.py`: GeneGAT 模型、预测和图特征提取。
- `src/gnns/train.py`: GeneGAT 训练循环、Focal Loss 和早停。

# 环境准备
## R
提前下载R，并把R_HOME加入系统变量
在 R 中先安装 BiocManager:`install.packages("BiocManager")`
使用 BiocManager 安装 limma `BiocManager::install("limma")`

## python
`pip install -r requirements.txt`
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
清华镜像源：pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple

# 推荐运行

```bash
python main.py --skip_shap
```

如果需要生成 SHAP 图：

```bash
python main.py
```
