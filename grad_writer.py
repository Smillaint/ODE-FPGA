import os
import glob

def write_grad_file(filepath, round_num, data_index_list, grad_list, mode='stream'):
    """
    每轮覆盖写（'w'），只保留本轮数据。
    C 程序只需要当前轮的梯度，无需保留历史轮。
    """
    with open(filepath, 'w') as f:          # ← 'a' → 'w'（覆盖写）
        f.write(f"Round {round_num}:\n")
        f.write("Initial:\n")

        for idx, grad in zip(data_index_list, grad_list):
            f.write(f"Data index {idx}:\n")
            for i, val in enumerate(grad):
                f.write(f"{val:.4e}")
                if i < len(grad) - 1:
                    f.write(",")
                if (i + 1) % 5 == 0:
                    f.write("\n")
            f.write(";\n")


def write_global_grad_file(filepath, round_num, global_grad):
    """
    每轮覆盖写（'w'），只保留本轮全局梯度。
    C 程序读取时只需当前轮的全局梯度。
    """
    with open(filepath, 'w') as f:          # ← 'a' → 'w'（覆盖写）
        f.write(f"Round {round_num}:\n")
        for i, val in enumerate(global_grad):
            f.write(f"{val:.4e}")
            if i < len(global_grad) - 1:
                f.write(",")
            if (i + 1) % 5 == 0:
                f.write("\n")
        f.write(";\n")


def clear_grad_files():
    for f in glob.glob("grad_client*.txt"):
        os.remove(f)
    if os.path.exists("global_grad.txt"):
        os.remove("global_grad.txt")
