# GeneGAT 项目速读

## 一句话

这个项目用乳腺癌基因表达数据做早晚期二分类。流程是：先筛选差异基因，再把每个病人样本表示成“基因为节点”的图，用 GAT 学习基因关系，最后比较原始表达特征、GAT 图嵌入特征、二者融合特征在 XGBoost 上的分类效果，并用 SHAP 做解释。

## 推荐阅读顺序

1. `main.py`
   命令行入口，参数集中在这里，适合先看整体实验怎么启动。

2. `src/pipline/runner.py`
   主流程编排。它把数据读取、建图、GAT 训练、XGBoost 对比、SHAP 解释串起来。

3. `src/gnns/data.py`
   读取 stage I-IV 表达矩阵，映射早晚期标签，调用基因筛选，标准化并划分 train / val / test。

4. `src/gnns/selecter.py`
   低表达过滤、方差过滤、limma 差异表达分析。这个文件负责“选哪些基因进入模型”。

5. `src/gnns/graph.py`
   构建基因图。它决定基因和基因之间怎么连，并把每个样本封装成 PyG Data。

6. `src/gnns/gat.py`
   GeneGAT 模型结构、预测、图嵌入提取。

7. `src/gnns/train.py`
   GAT 训练循环、Focal Loss、早停。

8. `src/pipline/xgboost.py`
   对 raw、graph、hybrid 三组特征训练 XGBoost。

9. `src/gnns/metrics.py`
   统一计算 Accuracy、F1、AUC。

10. `src/pipline/explain.py`
    SHAP 可解释性分析。

## 数据长什么样

原始输入是四个分期文件：

- `stagei_data.txt`
- `stageii_data.txt`
- `stageiii_data.txt`
- `stageiv_data.txt`

每个文件读进来以后都是“行是样本，列是基因”的表达矩阵。

```text
          GeneA   GeneB   GeneC   GeneD
样本1      2.1     0.5     7.3     1.2
样本2      1.8     0.7     6.9     2.0
样本3      5.2     3.1     1.4     8.8
```

标签映射：

- stage I 和 stage II 记为 `0`，表示早期。
- stage III 和 stage IV 记为 `1`，表示晚期。

所以模型做的是二分类：早期或晚期。

## 数据处理流程

1. `load_data`
   读取四个分期文件，把它们拼成一个总表达矩阵。

2. `select_data`
   根据配置选择基因。当前主流程使用 `method="lxs"`，优先读取 `data/deg_genes_fc0.27_p0.05.txt`。

3. `StandardScaler`
   对每个基因列做标准化，让不同基因表达值尺度更接近。

4. `split_once` 或 `split_fold`
   划分训练集、验证集、测试集。

5. 返回张量
   `prepare_data` 返回全量数据、训练/验证/测试数据、基因名、样本 ID 和 mask。

面试时可以强调：差异基因筛选和标准化是为了降低高维基因表达数据中的噪声和尺度差异。

## 图是怎么建的

这个项目的图不是“样本图”，而是“基因图”。

核心设定：

- 一个样本对应一张图。
- 图里的节点是基因。
- 节点特征是这个样本中该基因的表达值。
- 所有样本共用同一个 `edge_index`，也就是同一套基因关系。

例如筛选后有 100 个样本、60 个基因，那么数据会变成：

```text
100 张图
每张图有 60 个节点
每个节点有 1 个特征：当前样本里该基因的表达值
每张图有 1 个标签：早期或晚期
```

最好记的一句话：

```text
图结构描述基因之间怎么连；节点特征描述某个病人每个基因表达多少。
```

当前支持三种建边方式：

- `full_connected`
  所有基因两两相连。优点是简单，不需要先验；缺点是边很多。

- `correlation`
  按训练集里的基因表达相关性连边。只连接相关性高的基因。

- `knn`
  对每个基因找表达模式最相近的 k 个基因。

注意：`graph.py` 里构图只使用训练集信息，这是为了减少测试集信息泄漏。

## GeneGAT 模型细节

模型位置：`src/gnns/gat.py`

主要结构：

1. 输入投影
   每个基因节点最开始只有 1 维表达值，先通过线性层投影到 hidden dimension。

2. GATConv
   使用图注意力层学习基因邻居之间的重要性。注意力机制让模型可以自动判断哪些相连基因更关键。

3. 残差连接
   GAT 层之间保留残差，缓解深层训练时的信息衰减。

4. 软聚类池化
   模型学习每个基因属于不同 cluster 的软分配，然后把节点级表示聚合成图级表示。

5. 最大池化和最小池化拼接
   既保留强激活信号，也保留低响应信号。最后得到样本级图嵌入。

6. 分类器
   图嵌入经过降维和线性分类器，输出早期/晚期二分类 logits。

面试可以这样说：

“GAT 不是直接对表格做分类，而是先把每个样本转成基因图。模型通过注意力学习基因之间的关系，再通过软聚类池化得到样本级 embedding，最后用于分类和下游 XGBoost 融合。”

## 训练和评估

训练位置：`src/gnns/train.py`

关键点：

- 使用 `FocalLoss`
  适合类别不平衡任务，让模型更关注难分类样本。

- 使用 `class_weights=[1, 3]`
  当前设置里晚期类别权重更高，用于缓解类别不均衡。

- 使用早停
  同时观察验证集 loss 和 F1，避免小样本任务中过拟合。

- 使用梯度裁剪
  限制梯度范数，减少训练不稳定。

指标位置：`src/gnns/metrics.py`

统一输出：

- Accuracy
- F1
- AUC

## 为什么还接 XGBoost

GAT 训练完以后，会提取每个样本的图嵌入。然后比较三组特征：

- `raw`
  原始基因表达特征。

- `graph`
  GAT 学到的图嵌入特征。

- `hybrid`
  原始基因表达和图嵌入拼接。

这样可以做一个清楚的消融实验：

- raw 表示传统表达矩阵分类能力。
- graph 表示图神经网络学到的关系特征是否有用。
- hybrid 表示两者结合是否更强。

面试时可以说：这个设计兼顾了可解释性和关系建模能力。

## SHAP 怎么解释

SHAP 位置：`src/pipline/explain.py`

当前主要对 hybrid XGBoost 做解释，因为 hybrid 同时包含：

- 原始基因特征
- 图嵌入特征

如果 SHAP 排名前面的是原始基因，说明它可能是候选生物标志物。如果 SHAP 排名前面的是 `Emb_i`，说明 GAT 学到的关系模式对分类有贡献。

已有结果里比较靠前的原始基因包括：

- `CHPT1`
- `C8orf55`
- `HTR2B`
- `NHLH1`
- `LYZL2`

图嵌入里贡献较高的是：

- `Emb_1`
- `Emb_7`

## 常见面试问答

问：你的图节点是什么？

答：节点是基因，不是病人样本。每个样本是一张图，图上的每个节点对应一个基因。

问：不同样本的图一样吗？

答：图结构一样，所有样本共用同一个 `edge_index`。不同的是节点特征，也就是每个病人样本里的基因表达量。

问：为什么要先做差异基因筛选？

答：基因表达数据维度很高，样本量相对小。先筛掉低表达、低方差和不显著差异的基因，可以降低噪声和过拟合风险。

问：为什么用 GAT？

答：GAT 可以对邻居基因分配不同注意力权重，比普通图卷积更灵活。对于基因关系不确定的任务，注意力机制可以让模型自己学习哪些基因关系更重要。

问：为什么还要用 XGBoost？

答：GAT 提取的是关系表示，XGBoost 对小样本表格特征很强。比较 raw、graph、hybrid 可以判断图嵌入是否真的带来增益。

问：你的可解释性怎么做？

答：用 SHAP 分析 hybrid XGBoost。原始基因特征的 SHAP 值可以提供候选重要基因，图嵌入的 SHAP 值可以说明关系表示是否对分类有贡献。

## 风险点和改进方向

- 差异基因筛选最好严格只在训练集上 fit，再应用到验证集和测试集，进一步降低信息泄漏风险。
- 当前 full connected 图边数较多，基因数量变大时计算成本会上升。
- 可以系统比较 full connected、correlation、knn 三种图结构。
- 可以保存每次实验的参数配置，方便复现实验。
- 可以做多次随机种子或 K 折平均，减少单次划分偶然性。

## 运行方式

快速跑指标，不生成 SHAP：

```bash
python main.py --skip_shap
```

完整运行并生成 SHAP：

```bash
python main.py
```

加载已有模型：

```bash
python main.py --load_model models/experiment/model_fold_0.pt --skip_shap
```

主要输出：

- `results/experiment/metrics_summary.csv`
- `results/experiment/metrics_summary.md`
- `images/experiment/fold_0_hybrid_shap/`
- `models/experiment/model_fold_0.pt`
