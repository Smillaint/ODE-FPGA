import torch
import numpy as np


class FedDySSelector:
    """
    FedDyS: 基于动态重要性分数的数据选择
    重要性分数 = 样本损失值（高loss → 高信息量 → 优先选择）
    使用 EMA 跨轮更新历史分数
    """
    def __init__(self, dataset_size, ema_alpha=0.9):
        self.dataset_size  = dataset_size
        self.ema_alpha     = ema_alpha
        # 每个样本的历史重要性分数（初始化为 1.0）
        self.importance    = np.ones(dataset_size, dtype=np.float32)

    def compute_importance(self, model, data_loader, criterion, index_list):
        """
        计算当前轮次每个样本的重要性（基于 loss）
        返回：{全局索引: 重要性分数} 字典
        """
        model.eval()
        scores = {}
        with torch.no_grad():
            for (data, target), global_idx in zip(data_loader, index_list):
                output = model(data)
                loss   = criterion(output, target).item()
                scores[global_idx] = loss
        return scores

    def update_importance(self, scores):
        """
        用 EMA 更新历史重要性分数
        s_i ← α × s_i + (1-α) × loss_i
        """
        for global_idx, loss_val in scores.items():
            if global_idx < self.dataset_size:
                self.importance[global_idx] = (
                    self.ema_alpha * self.importance[global_idx]
                    + (1 - self.ema_alpha) * loss_val
                )

    def select(self, index_list, buffer_size):
        """
        从当前轮次的样本中选出 Top-K 重要样本
        """
        if len(index_list) <= buffer_size:
            return index_list  # 样本数不足，全选

        # 按历史重要性分数排序，选 Top-K
        scored = [
            (idx, self.importance[idx] if idx < self.dataset_size else 0.0)
            for idx in index_list
        ]
        scored.sort(key=lambda x: x[1], reverse=True)  # 高分优先
        selected = [idx for idx, _ in scored[:buffer_size]]

        print(f"    FedDyS 重要性分数范围: "
              f"[{scored[-1][1]:.4f}, {scored[0][1]:.4f}]")
        return selected
