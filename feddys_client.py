import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from model import get_grad_vector
from data_utils import get_stream_batch
from feddys_selector import FedDySSelector


class FedDySClient:
    def __init__(self, client_id, model, dataset,
                 lr=0.05, speed=100, buffer_size=20, local_epochs=5):
        self.client_id    = client_id
        self.model        = model
        self.dataset      = dataset
        self.lr           = lr
        self.speed        = speed
        self.buffer_size  = buffer_size
        self.local_epochs = local_epochs
        self.criterion    = nn.CrossEntropyLoss()
        self.optimizer    = torch.optim.SGD(model.parameters(), lr=lr)

        # FedDyS 选择器（维护全数据集大小的重要性向量）
        self.selector = FedDySSelector(
            dataset_size=len(dataset),
            ema_alpha=0.9
        )

    def _get_param_vector(self):
        return torch.cat([p.detach().clone().flatten()
                          for p in self.model.parameters()]).tolist()

    def train_one_round(self, round_num):
        """
        FedDyS 完整一轮流程：
        1. 获取当前流数据
        2. 计算各样本重要性（loss）
        3. 更新 EMA 历史分数
        4. 选 Top-K 样本
        5. 本地训练
        6. 返回参数差 Δθ
        """
        stream_data = get_stream_batch(self.dataset, round_num, self.speed)
        if stream_data is None:
            return None

        offset     = round_num * self.speed
        index_list = list(range(offset, offset + len(stream_data)))
        loader_all = DataLoader(stream_data, batch_size=1, shuffle=False)

        # ── Step 1: 计算重要性分数 ───────────────────────────
        scores = self.selector.compute_importance(
            self.model, loader_all, self.criterion, index_list
        )

        # ── Step 2: EMA 更新历史分数 ─────────────────────────
        self.selector.update_importance(scores)

        # ── Step 3: 选 Top-K 样本 ────────────────────────────
        selected_global = self.selector.select(index_list, self.buffer_size)
        selected_local  = [idx - offset for idx in selected_global]

        print(f"  Client {self.client_id}: "
              f"FedDyS 选中 {len(selected_local)} / {len(index_list)} 个样本")

        # ── Step 4: 本地训练 ─────────────────────────────────
        global_params   = self._get_param_vector()
        selected_subset = Subset(stream_data, selected_local)
        loader_train    = DataLoader(
            selected_subset,
            batch_size=min(32, len(selected_local)),
            shuffle=True
        )

        self.model.train()
        total_loss = 0.0
        for epoch in range(self.local_epochs):
            for data, target in loader_train:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss   = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        print(f"  Client {self.client_id}: "
              f"loss={total_loss:.4f} ({self.local_epochs} epochs)")

        local_params = self._get_param_vector()
        delta = [lp - gp for lp, gp in zip(local_params, global_params)]
        return delta
