import os
import torch
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import SimpleCNN
from client import FederatedClient
from server import FederatedServer
from data_utils import load_stream_data
from grad_writer import clear_grad_files
from ode_selector import call_c_selector

# ===== 超参数 =====
NUM_CLIENTS  = 3
NUM_ROUNDS   = 50          # ← 10 → 50
SPEED        = 100
BUFFER_SIZE  = 20          # ← 5 → 20，每轮选更多样本
LR           = 0.05        # ← 0.01 → 0.05，加大学习率
LOCAL_EPOCHS = 5           # ← 新增：本地训练轮数

BASE_DIR = r"C:\Users\31854\ODE_FL"


def main():
    clear_grad_files()

    client_data, test_set = load_stream_data('mnist', NUM_CLIENTS)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    server  = FederatedServer(SimpleCNN())
    clients = [
        FederatedClient(i, SimpleCNN(), client_data[i],
                        lr=LR, speed=SPEED, buffer_size=BUFFER_SIZE)
        for i in range(NUM_CLIENTS)
    ]
    results = []

    for round_num in range(NUM_ROUNDS):
        print(f"\n{'='*45}")
        print(f"  Round {round_num + 1} / {NUM_ROUNDS}")
        print(f"{'='*45}")

        client_grads = []

        for client in clients:
            print(f"\n[Client {client.client_id}]")
            client.model.load_state_dict(server.global_model.state_dict())

            # 1. 生成梯度并覆盖写文件
            grad_list, index_list = client.generate_grads_for_round(round_num)
            if not grad_list:
                print(f"  Client {client.client_id}: 数据已用完，跳过")
                continue

            # 2. 调用 C 选择器
            global_grad_path = os.path.join(BASE_DIR, "global_grad.txt")
            if round_num == 0 or not os.path.exists(global_grad_path):
                print(f"  [跳过 ODE 选择] global_grad.txt 尚未生成（第 0 轮）")
                raw_selected = []
            else:
                raw_selected = call_c_selector(
                    client.client_id,
                    round_num,
                    BUFFER_SIZE,
                    SPEED
                )

            # 3. 过滤出本 Client 本轮的索引
            local_selected = [idx for idx in raw_selected if idx in index_list]
            print(f"  ODE 选中 {len(local_selected)} / {len(index_list)} 个样本")

            if len(local_selected) == 0 and round_num > 0:
                print(f"  Client {client.client_id}: ODE 无本地命中，降级为全部样本")

            # 4. 本地训练（local_selected 为空时自动降级为全部）
            local_grad = client.train_on_selected(
                local_selected,
                round_num,
                index_list,
                local_epochs=LOCAL_EPOCHS    # ← 传入多轮本地训练
            )
            if local_grad:
                client_grads.append(local_grad)

        # 5. 服务器聚合（内部已写 global_grad.txt，无需重复写）
        if client_grads:
            server.aggregate(client_grads, round_num)
        else:
            print("\n  本轮无任何客户端参与训练，跳过聚合")

        # 6. 评估
        acc = server.evaluate(test_loader)
        results.append({'round': round_num + 1, 'accuracy': acc})
        print(f"\n>>> Round {round_num + 1} 精度: {acc:.2f}%")

    # ===== 保存结果 =====
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_ode.json", "w") as f:
        json.dump(results, f, indent=2)

    rounds = [r['round']    for r in results]
    accs   = [r['accuracy'] for r in results]
    plt.figure()
    plt.plot(rounds, accs, 'b-o')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.title('ODE-FL Baseline')
    plt.grid(True)
    plt.savefig("results/baseline_ode.png")
    print("\n✅ 实验完成！结果保存在 results/ 目录下")


if __name__ == "__main__":
    main()
