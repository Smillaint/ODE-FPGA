import numpy as np
import torch
from grad_writer import write_global_grad_file


class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model
        self.global_grad  = None

    def aggregate(self, client_grads, round_num):
        if not client_grads:
            print("  Server: 无梯度可聚合")
            return None

        # 1. 平均参数差
        self.global_grad = np.mean(client_grads, axis=0).tolist()
        delta_tensor = torch.tensor(self.global_grad, dtype=torch.float32)

        # 2. 原地更新全局模型参数
        pointer = 0
        for param in self.global_model.parameters():
            num         = param.numel()
            delta_chunk = delta_tensor[pointer:pointer + num].reshape(param.shape)
            param.data += delta_chunk
            pointer    += num

        # 3. 写全局梯度文件，用 round_num + 1 标记
        #    → 下一轮 C 选择器调用时传入的是 round_num+1，恰好对应
        write_global_grad_file("global_grad.txt", round_num + 1, self.global_grad)

        print(f"  Server 聚合完成，参数差 L2 norm = "
              f"{np.linalg.norm(self.global_grad):.6f}")
        first_param = next(self.global_model.parameters())
        print(f"  [DEBUG] 全局模型第一参数均值: {first_param.data.mean().item():.6f}")

        return self.global_grad

    def get_global_grad_vector(self):
        """返回最近一次聚合的全局梯度向量（供外部查询，server 内部已写文件）"""
        return self.global_grad

    def evaluate(self, test_loader):
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                output  = self.global_model(data)
                pred    = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total   += target.size(0)
        return 100.0 * correct / total
