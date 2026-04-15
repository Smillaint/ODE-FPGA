import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from model import get_grad_vector
from grad_writer import write_grad_file
from data_utils import get_stream_batch


class FederatedClient:
    def __init__(self, client_id, model, dataset, lr=0.01, speed=100, buffer_size=5):
        self.client_id   = client_id
        self.model       = model
        self.dataset     = dataset
        self.lr          = lr
        self.speed       = speed
        self.buffer_size = buffer_size
        self.criterion   = nn.CrossEntropyLoss()
        self.optimizer   = torch.optim.SGD(model.parameters(), lr=lr)

    # ── 提取当前模型参数向量 ─────────────────────────────────────────────
    def _get_param_vector(self):
        return torch.cat([p.detach().clone().flatten()
                          for p in self.model.parameters()]).tolist()

    # ── 从向量恢复模型参数 ───────────────────────────────────────────────
    def _set_param_vector(self, vec):
        tensor  = torch.tensor(vec, dtype=torch.float32)
        pointer = 0
        for param in self.model.parameters():
            num = param.numel()
            param.data.copy_(tensor[pointer:pointer + num].reshape(param.shape))
            pointer += num

    # ── 生成梯度文件（供 ODE C 程序读取）────────────────────────────────
    def generate_grads_for_round(self, round_num):
        stream_data = get_stream_batch(self.dataset, round_num, self.speed)
        if stream_data is None:
            return [], []

        model_backup = copy.deepcopy(self.model.state_dict())

        loader     = DataLoader(stream_data, batch_size=1, shuffle=False)
        grad_list  = []
        index_list = []

        self.model.eval()               # ← eval 模式，不污染 BN/Dropout
        for batch_idx, (data, target) in enumerate(loader):
            self.model.zero_grad()
            output     = self.model(data)
            loss       = self.criterion(output, target)
            loss.backward()
            grad       = get_grad_vector(self.model)
            global_idx = round_num * self.speed + batch_idx
            grad_list.append(grad)
            index_list.append(global_idx)

        self.model.load_state_dict(model_backup)
        self.model.zero_grad()

        write_grad_file(
            f"grad_client{self.client_id}.txt",
            round_num,
            index_list,
            grad_list,
        )
        return grad_list, index_list

    # ── 本地训练，返回参数差 Δθ（FedAvg 风格）──────────────────────────
    def train_on_selected(self, selected_indices, round_num, all_indices, local_epochs=5):
        # 无选中时降级为全部数据
        if not selected_indices:
            selected_indices = all_indices
            print(f"  Client {self.client_id}: 使用全部 {len(all_indices)} 个样本")
        else:
            print(f"  Client {self.client_id}: 使用 {len(selected_indices)} 个样本")

        stream_data = get_stream_batch(self.dataset, round_num, self.speed)
        if stream_data is None:
            print(f"  Client {self.client_id}: stream_data 为空，跳过")
            return None

        # 将全局索引转为本轮局部索引，过滤越界值
        stream_len    = len(stream_data)
        offset        = round_num * self.speed
        local_indices = [
            idx - offset
            for idx in selected_indices
            if 0 <= (idx - offset) < stream_len
        ]

        if not local_indices:
            print(f"  Client {self.client_id}: 无有效本地索引，跳过")
            return None

        global_params   = self._get_param_vector()
        selected_subset = Subset(stream_data, local_indices)
        loader          = DataLoader(
            selected_subset,
            batch_size=min(32, len(local_indices)),
            shuffle=True
        )

        self.model.train()
        total_loss = 0.0

        # ← 多轮本地训练（local_epochs 控制）
        for epoch in range(local_epochs):
            for data, target in loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss   = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        print(f"  Client {self.client_id}: loss={total_loss:.4f} ({local_epochs} epochs × {len(local_indices)} 样本)")

        # 返回参数差 Δθ = θ_local - θ_global
        local_params = self._get_param_vector()
        delta = [lp - gp for lp, gp in zip(local_params, global_params)]
        return delta
