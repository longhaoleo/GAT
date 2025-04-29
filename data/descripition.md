# 数据说明

下载

`TCGA_BRCA_TPMed.csv` 数据包含乳腺癌样本的基因表达数据（**BRCA**乳腺浸润癌）
它是一个 TPM（Transcripts Per Million） 归一化的基因表达矩阵。
每一行都是不同的基因，每一列是不同的样本
`TCGA.csv` 数据经过`data_deleter.py`的筛选，选出79个需要的基因

在 TCGA 数据集中，样本 ID 通常遵循 `TCGA-XX-YYYY-ZZ` 这种格式，其中：
其中ZZ编号01~09表示肿瘤，10~19表示正常对照

TCGA_BRCA_TPM.csv是老师发的，经过某种预处理。

# TPM（Transcripts Per Million）的含义
TPM 是 RNA-seq 数据的一种标准化表达值，表示一个基因在一个样本中的转录本数量，单位是百万分之一（per million）。TPM 解决了不同基因长度和不同样本测序深度对基因表达量的影响，使得不同样本之间的基因表达量可比。

# TPM 计算方法
TPM 通过以下步骤计算得到：
1. 计算 Reads Per Kilobase（RPK）
- RNA-seq 读取的数据是 reads count（即测到多少个片段与该基因匹配）。
- RPK = reads count / 基因长度（kb）
这样做的目的是消除基因长度的影响，因为较长的基因通常会比较短的基因积累更多的 reads。

2. 计算 RPK 总和
对每个样本，所有基因的 RPK 加总，得到 RPK_sum。
3. 计算 TPM
TPM = (RPK / RPK_sum) × 1,000,000
这样做的目的是消除测序深度的影响，使得所有样本的 TPM 总和都是 100 万（1,000,000）。

# 总结
TPM 代表基因在样本中的表达水平，已归一化，可以用于样本间比较。
高 TPM 代表该基因在该样本中表达量较高，低 TPM 代表表达量低。
不同样本的 TPM 直接可比，不同基因的 TPM 不能直接比较。