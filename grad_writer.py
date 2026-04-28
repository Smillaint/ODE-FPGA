import os
from config import ensure_grad_dir, grad_path


def write_grad_file(filename, round_num, index_list, grad_list):
    ensure_grad_dir()
    path = grad_path(filename)
    with open(path, 'a') as f:
        f.write(f"Round {round_num}\n")
        f.write("Initial:\n")
        for idx, grad in zip(index_list, grad_list):
            f.write(f"Data index {idx}\n")
            f.write(",".join(f"{g:.8f}" for g in grad) + ",\n")
    print(f"  [GradWriter] 写出 {path}，共 {len(index_list)} 条")


def write_global_grad_file(filename, round_num, grad_vector):
    ensure_grad_dir()
    path = grad_path(filename)
    with open(path, 'a') as f:
        f.write(f"Round {round_num}\n")
        f.write(",".join(f"{g:.8f}" for g in grad_vector) + ",\n")
    print(f"  [GradWriter] 全局梯度写出 {path}，Round={round_num}")


def clear_grad_files():
    for name in ["global_grad.txt"] + [f"grad_client{i}.txt" for i in range(10)]:
        p = grad_path(name)
        if os.path.exists(p):
            os.remove(p)
            print(f"  [GradWriter] 清除旧文件: {p}")
