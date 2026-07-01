import argparse
import sys

# 让根目录下的 main.py 可以直接导入 src/ 里的项目模块。
sys.path.append("src")


def parse_args():
    """集中管理实验参数，默认值就是当前项目采用的一套配置。"""
    parser = argparse.ArgumentParser(description="GeneGAT 乳腺癌分期预测实验")

    # 输出目录：训练好的模型、指标表、SHAP 图都会保存到这里。
    parser.add_argument("--model_dir", type=str, default="models/experiment")
    parser.add_argument("--results_dir", type=str, default="results/experiment")
    parser.add_argument("--image_dir", type=str, default="images/experiment")

    # 复现实验时可以加载旧模型；为空时会重新训练 GAT。
    parser.add_argument("--load_model", type=str, default="")

    # 数据和图结构设置：
    # full_connected 更简单直接；correlation/knn 更强调基因间关系。
    parser.add_argument("--k_fold", type=int, default=1)
    parser.add_argument(
        "--graph_method",
        type=str,
        default="full_connected",
        choices=["full_connected", "correlation", "knn"],
    )

    # GAT 训练参数：小样本任务里主要关注过拟合，所以保留早停和 dropout。
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)

    # GAT 模型结构参数：节点是基因，模型输出样本级图嵌入。
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--pooling_dim", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--gat_layers", type=int, default=2)
    parser.add_argument("--activation", type=str, default="swish")
    parser.add_argument("--cluster_num", type=int, default=8)

    # 跳过 SHAP 可以更快看到训练和分类指标。
    parser.add_argument("--skip_shap", action="store_true")

    return parser.parse_args()


def main():
    # 主流程在 src/pipline/runner.py：
    # 1. 读取并筛选基因表达数据
    # 2. 构建基因图并训练 GeneGAT
    # 3. 提取图嵌入，比较 raw/graph/hybrid 三种 XGBoost
    # 4. 保存指标表，并可选生成 SHAP 解释图
    args = parse_args()

    # 放在这里导入，是为了让 `python main.py --help` 不依赖 torch 等训练环境。
    from pipline.runner import run_pipeline

    run_pipeline(args)


if __name__ == "__main__":
    main()
