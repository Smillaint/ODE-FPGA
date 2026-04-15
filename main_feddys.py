import os
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import SimpleCNN
from server import FederatedServer        # ← 复用 ODE 的 server
from feddys_client import FedDySClient
from data_utils import load_stream_data

# ===== 超参数（与 ODE 保持一致，便于公平对比）=====
NUM_CLIENTS  = 3
NUM_ROUNDS   = 50
SPEED        = 100
BUFFER_SIZE  = 20
LR           = 0.05
LR_MIN       = 0.005
LOCAL_EPOCHS = 5


def main():
    client_data, test_set = load_stream_data('mnist', NUM_CLIENTS)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    server  = FederatedServer(SimpleCNN())
    clients = [
        FedDySClient(i, SimpleCNN(), client_data[i],
                     lr=LR, speed=SPEED,
                     buffer_size=BUFFER_SIZE,
                     local_epochs=LOCAL_EPOCHS)
        for i in range(NUM_CLIENTS)
    ]
    results = []

    for round_num in range(NUM_ROUNDS):
        print(f"\n{'='*45}")
        print(f"  FedDyS Round {round_num + 1} / {NUM_ROUNDS}")
        print(f"{'='*45}")

        client_grads = []
        for client in clients:
            # 同步全局模型
            client.model.load_state_dict(server.global_model.state_dict())

            delta = client.train_one_round(round_num)
            if delta:
                client_grads.append(delta)

        if client_grads:
            server.aggregate(client_grads, round_num)

        # 学习率线性衰减
        progress   = round_num / max(NUM_ROUNDS - 1, 1)
        current_lr = LR - (LR - LR_MIN) * progress
        for client in clients:
            for pg in client.optimizer.param_groups:
                pg['lr'] = current_lr

        acc = server.evaluate(test_loader)
        results.append({'round': round_num + 1, 'accuracy': acc})
        print(f"\n>>> FedDyS Round {round_num + 1} 精度: {acc:.2f}%")

    # ===== 保存结果 =====
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_feddys.json", "w") as f:
        json.dump(results, f, indent=2)

    rounds = [r['round']    for r in results]
    accs   = [r['accuracy'] for r in results]
    plt.figure()
    plt.plot(rounds, accs, 'r-s')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.title('FedDyS Baseline')
    plt.grid(True)
    plt.savefig("results/baseline_feddys.png")
    print("\n✅ FedDyS 实验完成！结果保存在 results/ 目录下")


if __name__ == "__main__":
    main()
