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

## 模型输入的变化过程

这一节按“数据在代码里一步步变成什么”来理解。

第一步：原始表达矩阵。

```text
X_raw: [样本数, 原始基因数]
y:     [样本数]
```

每一行是一个病人样本，每一列是一个基因表达值。

第二步：基因筛选。

```text
X_selected: [样本数, 筛选后基因数]
```

`selecter.py` 会先做低表达过滤、方差过滤和差异表达分析。筛选后基因数会减少，目的是降低噪声和过拟合风险。这里的“筛选后基因数”就是后面图里的节点数量。

第三步：标准化。

```text
X_scaled: [样本数, 筛选后基因数]
```

`StandardScaler` 按列处理，也就是对每个基因分别做标准化。这样不同基因的表达值尺度更一致，避免某些数值范围大的基因主导训练。

第四步：划分数据集。

```text
X_train: [训练样本数, 筛选后基因数]
X_val:   [验证样本数, 筛选后基因数]
X_test:  [测试样本数, 筛选后基因数]
```

第五步：构建共享基因图。

```text
edge_index: [2, 边数]
```

`edge_index` 描述“基因节点之间怎么连”。当前默认是 `full_connected`，也就是筛选后的所有基因两两可交互。

第六步：把每个样本变成一张图。

对于第 `i` 个样本：

```text
原始一行表达向量:       [筛选后基因数]
转成节点特征矩阵 x:     [筛选后基因数, 1]
共享边 edge_index:      [2, 边数]
图标签 y_i:             [1]
```

也就是说，`create_data_objects` 会把表格中的每一行变成一个 PyG `Data`：

```text
Data(
    x = 当前样本的基因表达,      shape = [G, 1]
    edge_index = 共享基因图,     shape = [2, E]
    y = 当前样本早/晚期标签
)
```

第七步：DataLoader 组成 batch。

如果 batch size 是 `B`，筛选后基因数是 `G`：

```text
batch.x:          [B * G, 1]
batch.edge_index: 多张图合并后的边索引
batch.batch:      [B * G]，标记每个节点属于哪张样本图
```

这就是 GeneGAT 真正收到的输入。

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

## 全连接图和注意力怎么理解

当前默认图结构是 `full_connected`，也就是同一张样本图中每个基因都可以和所有其他基因交互。

这和 Transformer 有相似之处：

- Transformer 里每个 token 可以 attend 到其他 token。
- 这里每个基因节点可以 attend 到其他基因节点。
- 全连接边提供“可交互范围”，GAT attention 学习“实际更重视谁”。

但它不是把所有病人样本混成一张大图。更准确的理解是：

```text
每个样本各自是一张基因图
所有样本共用同一个基因拓扑 edge_index
所有样本共用同一套 GAT 参数
每个样本因为表达值不同，会产生不同的注意力响应和图嵌入
```

所以可以把它理解成一个“共享的基因注意力场”：

- 共享的是基因位置、基因连接方式、GAT 参数。
- 不共享的是每个病人的节点表达值。
- 模型学习的是在这个共享基因空间里，哪些基因关系模式和早晚期分类有关。

## GeneGAT 模型细节

模型位置：`src/gnns/gat.py`

默认模型参数来自 `main.py`：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `input_dim` | `1` | 每个基因节点的输入特征维度，当前只有该样本中的表达值 |
| `output_dim` | `2` | 二分类输出，早期和晚期 |
| `hidden_dim` | `128` | 节点隐藏表示维度 |
| `num_heads` | `8` | GAT 多头注意力头数 |
| `gat_layers` | `2` | GATConv 层数 |
| `dropout` | `0.5` | 防止小样本过拟合 |
| `activation` | `swish` | 激活函数，对应 PyTorch 的 `SiLU` |
| `cluster_num` | `8` | 软聚类池化中的 cluster 数 |
| `pooling_dim` | `8` | 图级 embedding 最终降维后的维度 |

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

## 图模型的架构设计方式

GeneGAT 的设计可以按“节点表示学习、图级聚合、样本分类”三层理解。

第一层：节点表示学习。

```text
输入节点特征 [G, 1]
经过 Linear(1, 128)
得到每个基因节点的隐藏表示 [G, 128]
```

为什么需要这一步：

- 原始表达值只是一个标量，表达能力有限。
- 投影到 128 维后，每个基因节点有更丰富的隐空间表示。
- 这类似 NLP 里把 token 映射成 embedding，只不过这里 token 是基因，输入值是表达量。

第二层：图注意力传播。

```text
GATConv 在 edge_index 定义的边上做信息传递
每个基因聚合邻居基因信息
注意力权重决定不同邻居的重要性
```

在全连接图下，这一步可以理解为“每个基因都可以看所有其他基因，但模型自己学哪些关系更重要”。这也是它和普通全连接 MLP 的区别：MLP 直接混合特征，GAT 则显式以“节点和边”的形式做关系建模。

第三层：残差和归一化。

```text
输入投影后使用 LayerNorm
GAT 层之间使用 residual connection
每层后使用 activation 和 dropout
```

这些设计主要是为了稳定训练。小样本基因任务很容易过拟合，所以 `dropout=0.5` 比较大。

第四层：软聚类池化。

```text
cluster_assignment: [G, 128] -> [G, 8]
```

它不是简单把所有基因平均，而是学习每个基因属于 8 个 cluster 的概率。可以把 cluster 理解成模型自动学出的“基因功能组合”或“表达模式组”。

对每张图：

```text
graph_clusters = assignment.T @ node_features
shape: [8, 128]
```

第五层：图级表示。

```text
max_pool(graph_clusters): [128]
min_pool(graph_clusters): [128]
concat:                   [256]
dim_reduction:             [8]
```

最后这个 8 维向量就是样本级 graph embedding。它既用于 GAT 自己分类，也会作为 `graph` 特征输入 XGBoost。

第六层：分类输出。

```text
classifier: Linear(8, 2) + Dropout
输出: [早期logit, 晚期logit]
```

## GeneGAT 维度流转

假设筛选后有 `G` 个基因，一个 batch 里有 `B` 个样本。

| 阶段                  | 张量形状           | 说明                      |
| ------------------- | -------------- | ----------------------- |
| 单个样本输入              | `[G, 1]`       | 每个基因一个表达值               |
| PyG batch 后         | `[B * G, 1]`   | 多张样本图合成一个 batch         |
| 输入投影后               | `[B * G, 128]` | `Linear(1, hidden_dim)` |
| GAT 层后              | `[B * G, 128]` | 多头注意力更新每个基因节点表示         |
| cluster assignment  | `[B * G, 8]`   | 每个基因软分配到 8 个 cluster    |
| 每张图的 cluster 特征     | `[8, 128]`     | 对同一张图内节点做 `A.T @ X`     |
| max/min pooling 拼接  | `[256]`        | 最大池化 128 维，最小池化 128 维   |
| 降维后 graph embedding | `[8]`          | `pooling_dim=8`         |
| 分类输出                | `[2]`          | 早期/晚期 logits            |

这里最重要的是：GAT 先学节点级基因表示，软聚类池化再把一张基因图压成一个样本级表示。

## 关键模块解释

输入投影：

- 原始节点只有 1 维表达值，太低维。
- `input_proj` 把表达值映射到 128 维，让后续 GAT 有足够表达空间。
- `LayerNorm` 用来稳定训练。

GATConv：

- 当前默认 2 层。
- 第一层和中间层使用多头 concat，最后一层保持输出维度为 128。
- 注意力权重可以理解为“某个基因在聚合邻居基因信息时，更重视哪些邻居”。

残差连接：

- 每层 GAT 后叠加一个线性残差。
- 作用是缓解过平滑和梯度传播问题，尤其是图层数增加时更有用。

软聚类池化：

- `cluster_assignment` 输出每个节点属于 8 个 cluster 的概率。
- 对一张图内部，使用 `A.T @ X` 得到 cluster 级表示。
- 这比简单 mean pooling 更灵活，因为模型可以学习“哪些基因组合成一类功能模式”。

max/min pooling：

- max pooling 保留最强响应。
- min pooling 保留低响应或抑制模式。
- 拼接后维度是 `128 * 2 = 256`。

降维和分类：

- `dim_reduction` 把 256 维压到 8 维。
- 这 8 维就是后续 XGBoost 中的 `graph` 特征。
- 分类器输出 2 维 logits，用于 GAT 自身二分类。

## 训练和评估

训练位置：`src/gnns/train.py`

默认训练参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `batch_size` | `32` | 每次训练 32 张样本图 |
| `epochs` | `300` | 最大训练轮数 |
| `patience` | `100` | 验证集长期不提升时早停 |
| `lr` | `1e-4` | AdamW 学习率 |
| `weight_decay` | `1e-4` | 权重衰减，抑制过拟合 |
| `dropout` | `0.5` | GAT 和分类器中的随机失活 |
| `class_weights` | `[1, 3]` | 晚期类别权重更高 |
| `threshold` | `0.5` | 二分类概率阈值 |
| 梯度裁剪 | `max_norm=1.0` | 避免梯度爆炸 |

关键点：

- 使用 `FocalLoss`
  适合类别不平衡任务，让模型更关注难分类样本。

  形式上可以理解为：普通交叉熵会被 `(1 - pt)^gamma` 调整。模型越容易分对的样本，损失权重越小；难分样本权重更大。

- 使用 `class_weights=[1, 3]`
  当前设置里晚期类别权重更高，用于缓解类别不均衡。

- 使用早停
  同时观察验证集 loss 和 F1，避免小样本任务中过拟合。

- 使用梯度裁剪
  限制梯度范数，减少训练不稳定。

训练时保存的是验证集表现最好的 `state_dict`，最后再加载回来。这样最终用于测试和特征提取的是验证集上最优的一版 GAT。

指标位置：`src/gnns/metrics.py`

统一输出：

- Accuracy
- F1
- AUC

## 具体处理操作流程

这里按 `src/pipline/runner.py` 的真实执行顺序看。

1. 创建输出目录
   `models/experiment` 保存模型，`results/experiment` 保存指标，`images/experiment` 保存 SHAP 图。

2. 读取和准备数据
   调用 `prepare_data`，得到 `X_train`、`X_val`、`X_test`、标签和基因名。

3. 构建基因图
   调用 `create_graph(X_train, method=args.graph_method)`，只用训练集估计图结构，避免测试集泄漏。

4. 创建 PyG DataLoader
   train、val、test 都复用同一个 `edge_index`，但是每个 split 里的样本表达值不同。

5. 创建 GeneGAT
   `build_model` 使用命令行参数创建模型，默认是 2 层 GAT、hidden 128、8 个 attention heads。

6. 训练或加载 GAT
   如果指定 `--load_model`，直接加载；否则训练并保存 `model_fold_0.pt`。

7. 测试 GAT
   调用 `predict_with_loader`，输出 GAT 自身的 Accuracy、F1、AUC。

8. 提取图嵌入
   调用 `extract_pooling_features`，得到 train、val、test 的 graph embedding。

9. 构造三组 XGBoost 特征
   `raw` 是原始表达矩阵，`graph` 是 GAT 嵌入，`hybrid` 是二者拼接。

10. 训练三组 XGBoost
    分别训练 raw-XGBoost、graph-XGBoost、hybrid-XGBoost，用同一套默认参数做公平对比。

11. 可选 SHAP
    如果不加 `--skip_shap`，对 hybrid XGBoost 做 SHAP，保存重要特征图和表格。

12. 保存汇总结果
    写出 `metrics_summary.csv` 和 `metrics_summary.md`。

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

## XGBoost 参数

XGBoost 默认参数在 `src/pipline/xgboost.py` 的 `DEFAULT_XGB_PARAMS`：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `objective` | `binary:logistic` | 二分类概率输出 |
| `eval_metric` | `logloss` | 训练时使用 log loss 作为评估项 |
| `n_estimators` | `100` | 树的数量 |
| `max_depth` | `9` | 单棵树最大深度 |
| `learning_rate` | `0.1` | 每棵树贡献的步长 |
| `gamma` | `0.2` | 分裂所需的最小损失下降 |
| `colsample_bytree` | `0.6` | 每棵树采样 60% 特征 |
| `subsample` | `1.0` | 使用全部样本 |
| `random_state` | `42` | 保证可复现 |

这里的参数不是大规模调参结果，更像是固定一组配置做公平对比。因为项目核心不是把 XGBoost 调到极致，而是比较 raw、graph、hybrid 三种特征是否有差异。

## 图构建参数

图构建在 `src/gnns/graph.py`：

| 方法 | 当前参数 | 含义 |
| --- | --- | --- |
| `full_connected` | 默认入口使用 | 所有基因两两相连 |
| `correlation` | `threshold=0.6` | 只保留绝对相关系数大于 0.6 的边 |
| `knn` | `k=15` | 每个基因连接表达模式最接近的 15 个基因 |

当前 `main.py` 默认使用 `full_connected`。面试时可以说：默认全连接是为了让 GAT 自己通过注意力学习边的重要性；后续可以比较 correlation 和 knn 来验证图先验是否带来提升。

## 后续实验流程安排

如果继续做实验，可以按下面顺序推进。

第一组：确认主结果。

- 固定 `graph_method=full_connected`
- 跑 `raw`、`graph`、`hybrid`
- 记录 GAT、raw-XGBoost、graph-XGBoost、hybrid-XGBoost 的 Accuracy、F1、AUC

目的：确认 hybrid 是否优于 raw，以及 graph embedding 是否提供额外信息。

第二组：比较图结构。

- 跑 `full_connected`
- 跑 `correlation`
- 跑 `knn`

目的：回答“全连接是否必要，基于相关性或 KNN 的基因先验是否更好”。

第三组：模型参数消融。

- `hidden_dim`: 32、64、128
- `pooling_dim`: 8、16、32
- `gat_layers`: 1、2、3
- `dropout`: 0.3、0.5、0.7

目的：看当前 1 维到 128 维是否过强，模型是否过拟合。

第四组：训练稳定性实验。

- 多随机种子重复
- K 折平均
- 记录均值和标准差

目的：减少单次划分偶然性，让结果更可靠。

第五组：解释性分析。

- 只对表现最好的 hybrid 模型做 SHAP
- 分别看原始基因特征和 `Emb_i` 图嵌入特征
- 汇总稳定出现的重要基因

目的：把模型结果转成可解释的候选生物标志物和关系模式。

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

## 参数调节思路

如果模型过拟合：

- 增大 `dropout`
- 增大 `weight_decay`
- 减小 `hidden_dim`
- 减小 `max_depth`
- 使用更严格的早停

如果模型欠拟合：

- 增大 `hidden_dim`
- 增加 `gat_layers`
- 增大 `pooling_dim`
- 降低 `dropout`
- 让 XGBoost 使用更多树或更大深度

如果训练不稳定：

- 降低 `lr`
- 保留梯度裁剪
- 检查边索引是否越界
- 检查类别是否极度不平衡

如果想验证图结构是否真的有用：

- 比较 `full_connected`、`correlation`、`knn`
- 比较 `raw` 和 `graph`
- 比较 `raw` 和 `hybrid`
- 看 SHAP 中 `Emb_i` 是否有稳定贡献

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
