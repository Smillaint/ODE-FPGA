import subprocess
import os
from config import C_SELECTOR_EXE, HLS_CSIM_EXE, grad_path


def call_c_selector(client_id, round_num, buffer_size, speed, use_fpga=True):
    """
    统一入口：决定调用 HLS 仿真还是 C 编译的基准程序
    """
    if use_fpga and HLS_CSIM_EXE and os.path.exists(HLS_CSIM_EXE):
        print("  [ODE] 🔷 使用 FPGA 仿真选择器")
        indices = _call_fpga_selector(client_id, round_num, buffer_size, speed)
        if indices is not None:
            return indices
        print("  [ODE] ⚠️ FPGA 失败，回退 C 选择器")

    print("  [ODE] 🔶 使用 C 选择器")
    return _call_c_selector(client_id, round_num, buffer_size, speed)

def _call_fpga_selector(client_id, round_num, buffer_size, speed):
    """
    调用 FPGA HLS 仿真的 csim.exe
    """
    grad_file = grad_path(f"grad_client{client_id}.txt")
    global_grad_file = grad_path("global_grad.txt")
    selected_file = grad_path("selected_indices.txt")

    cmd = [
        str(HLS_CSIM_EXE),
        str(grad_file),
        str(global_grad_file),
        str(round_num),
        str(buffer_size),
        str(speed)
    ]

    try:
        # 核心修复：增加 encoding 和 errors 参数，防止 GBK 解码报错
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=120
        )

        if result.returncode != 0:
            return None

        return _parse_selected_file(selected_file)
    except Exception as e:
        print(f"  [FPGA Error] {e}")
        return None

def _call_c_selector(client_id, round_num, buffer_size, speed):
    """
    调用传统的 C 语言编译的选择器 (selector.exe)
    """
    grad_file = grad_path(f"grad_client{client_id}.txt")
    global_grad_file = grad_path("global_grad.txt")
    selected_file = grad_path("selected_indices.txt")

    if not os.path.exists(C_SELECTOR_EXE):
        print(f"  [Error] 找不到 C 选择器: {C_SELECTOR_EXE}")
        return []

    cmd = [
        str(C_SELECTOR_EXE),
        str(grad_file),
        str(global_grad_file),
        str(round_num),
        str(buffer_size),
        str(speed),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return _parse_selected_file(selected_file)
    except Exception as e:
        print(f"  [C Error] {e}")
        return []

def _parse_selected_file(file_path):
    """
    解析 selected_indices.txt 获取索引列表
    """
    indices = []
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line or "Round" in line:
                    continue
                indices.append(int(line))
        return indices
    except Exception as e:
        print(f"  [Parse Error] {e}")
        return None
