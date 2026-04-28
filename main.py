import os
import torch
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import SimpleCNN
from client import FederatedClient
from server import FederatedServer
from data_utils import load_stream_data
from config import grad_path
from grad_writer import clear_grad_files
from ode_selector import call_c_selector

NUM_CLIENTS  = 3
NUM_ROUNDS   = 50
SPEED        = 100
BUFFER_SIZE  = 20
LR           = 0.05
LOCAL_EPOCHS = 5


def csd_delete_unselected(client_id, round_num, selected_indices, speed):
    start_idx   = round_num * speed
    end_idx     = start_idx + speed
    all_indices = set(range(start_idx, end_idx))
    selected    = set(selected_indices)
    deleted_cnt = len(all_indices - selected)
    print(f"  [CSD] Client {client_id}: "
          f"保留 {len(selected)} 条，模拟丢弃 {deleted_cnt} 条低价值数据")
    return list(selected)


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
        print(f"\n{'='*50}")
        print(f"  Round {round_num + 1} / {NUM_ROUNDS}")
        print(f"{'='*50}")

        client_grads = []

        for client in clients:
            print(f"\n[Client {client.client_id}]")
            client.model.load_state_dict(server.global_model.state_dict())

            grad_list, index_list = client.generate_grads_for_round(round_num)
            if not grad_list:
                print(f"  Client {client.client_id}: 数据已用完，跳过")
                continue

            # ← 路径直接用 ODE_FL_DIR
            global_grad_path = grad_path("global_grad.txt")
            if round_num == 0 or not os.path.exists(global_grad_path):
                print("  [跳过 ODE 选择] global_grad.txt 尚未生成（第 0 轮）")
                raw_selected = []
            else:
                raw_selected = call_c_selector(
                    client.client_id, round_num, BUFFER_SIZE, SPEED
                )

            local_selected = [idx for idx in raw_selected if idx in index_list]
            print(f"  ODE 选中 {len(local_selected)} / {len(index_list)} 个样本")

            if round_num > 0 and local_selected:
                local_selected = csd_delete_unselected(
                    client.client_id, round_num, local_selected, SPEED
                )

            local_grad = client.train_on_selected(
                local_selected, round_num, index_list,
                local_epochs=LOCAL_EPOCHS
            )
            if local_grad:
                client_grads.append(local_grad)

        if client_grads:
            server.aggregate(client_grads, round_num)
        else:
            print("\n  本轮无任何客户端参与训练，跳过聚合")

        acc = server.evaluate(test_loader)
        results.append({'round': round_num + 1, 'accuracy': acc})
        print(f"\n>>> Round {round_num + 1} 精度: {acc:.2f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_ode.json", "w") as f:
        json.dump(results, f, indent=2)

    rounds = [r['round']    for r in results]
    accs   = [r['accuracy'] for r in results]
    plt.figure()
    plt.plot(rounds, accs, 'b-o')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.title('ODE-FL + CSD Acceleration')
    plt.grid(True)
    plt.savefig("results/baseline_ode.png")
    print("\n✅ 实验完成！结果保存在 results/ 目录下")


if __name__ == "__main__":
    main()
