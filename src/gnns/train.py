"""
GeneGAT 训练模块。

gat.py 只保留模型结构、预测和特征提取；
这个文件专门放训练相关代码：
- FocalLoss
- train_graph_classifier
- 训练过程中的梯度检查工具
"""

import copy
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch_geometric.data import Data, DataLoader


def train_graph_classifier(
    model,
    train_data,
    val_data=None,
    batch_size=32,
    epochs=100,
    lr=0.001,
    weight_decay=1e-5,
    patience=15,
    class_weights=None,
    num_workers=0,
):
    """
    训练图分类模型，支持直接传入数据集或 DataLoader。

    参数:
        model: PyTorch Geometric 模型实例
        train_data: 训练数据集或训练 DataLoader
        val_data: 验证数据集或验证 DataLoader
        batch_size: 批处理大小，当传入数据集时使用
        epochs: 最大训练轮数
        lr: 学习率
        weight_decay: AdamW 的权重衰减
        patience: 早停耐心值
        class_weights: 类别权重，例如 [1, 3] 表示类别 1 权重更高
        num_workers: DataLoader 工作线程数

    返回:
        model: 训练好的模型
        threshold: 当前固定使用的二分类阈值
    """
    if not isinstance(train_data, DataLoader):
        print(f"将数据集转换为DataLoader，批处理大小: {batch_size}")
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        train_loader = train_data
        val_loader = val_data
        print(f"使用提供的DataLoader: 训练批次数={len(train_loader)}, 验证批次数={len(val_loader)}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if class_weights is not None:
        alpha = torch.tensor(class_weights, dtype=torch.float)
        print("alpha shape:", alpha.shape)
        print("alpha:", alpha)
        if torch.isnan(alpha).any() or torch.isinf(alpha).any():
            print("Alpha contains NaN or Inf!")
            alpha = torch.nan_to_num(alpha)
        criterion = FocalLoss(alpha=alpha, gamma=2)
        print("Using weighted loss.")
    else:
        criterion = FocalLoss(gamma=2)
        print("Using unweighted Focal loss.")

    best_model_state = None
    epochs_no_improve = 0
    threshold = 0.5
    best_f1 = 0.0
    best_val_loss = float("inf")

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        model.train()
        total_loss = 0
        correct_train = 0
        total_train_samples = 0
        optimizer.zero_grad()

        for batch in train_loader:
            if batch.edge_index.numel() > 0:
                max_index = torch.max(batch.edge_index)
                num_nodes = batch.x.size(0)
                if num_nodes <= max_index:
                    print(f"警告: 批次的节点数 ({num_nodes}) 小于或等于边索引中的最大值 ({max_index}).")
                    continue

            logits = model(batch)
            target = batch.y.squeeze().long()
            loss = criterion(logits, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct_train += (pred == target).sum().item()
            total_train_samples += batch.num_graphs

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train_samples
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val_samples = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch)
                target = batch.y.squeeze().long()
                loss = criterion(logits, target)
                total_val_loss += loss.item()

                probs = torch.softmax(logits, dim=1)
                pred = (probs[:, 1] > threshold).int()

                all_val_preds.append(pred)
                all_val_targets.append(target)
                correct_val += (pred == target).sum().item()
                total_val_samples += batch.num_graphs

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val / total_val_samples

        all_val_preds = torch.cat(all_val_preds).cpu().numpy()
        all_val_targets = torch.cat(all_val_targets).cpu().numpy()
        current_f1 = f1_score(all_val_targets, all_val_preds, average="binary")

        epoch_duration = time.time() - epoch_start_time
        print(
            f"Epoch {epoch}/{epochs} [{epoch_duration:.2f}s]: "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"Val F1: {current_f1:.4f}"
        )

        improved = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            improved = True
            print(f"验证损失提升到 {best_val_loss:.4f}")

        if current_f1 > best_f1:
            best_f1 = current_f1
            improved = True
            print(f"F1分数提升到 {best_f1:.4f}")

        if improved:
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print("模型状态已保存。")
        else:
            epochs_no_improve += 1
            print(f"验证指标未提升 {epochs_no_improve} 轮。")
            if epochs_no_improve >= patience:
                print(f"早停在第 {epoch} 轮触发。")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"加载最佳模型状态，验证损失: {best_val_loss:.4f}, F1分数: {best_f1:.4f}")
    else:
        print("训练完成，但没有改进或早停未触发。")

    return model, threshold


class FocalLoss(nn.Module):
    """适合类别不平衡任务的 Focal Loss。"""

    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (list, tuple, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        else:
            self.alpha = alpha

        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, targets):
        targets = targets.long()
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)

        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha.to(inputs.device)[targets]
        else:
            alpha_t = self.alpha

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def check_gradients(model, step=0, log_interval=100):
    """训练调试工具：定期检查是否有参数没有梯度。"""
    if step % log_interval != 0:
        return

    zero_grad_params = []
    none_grad_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                none_grad_params.append(name)
            elif torch.all(param.grad == 0):
                zero_grad_params.append(name)

    if none_grad_params:
        print(f"警告: 以下参数的梯度为None: {none_grad_params}")
    if zero_grad_params:
        print(f"警告: 以下参数的梯度全为0: {zero_grad_params}")
