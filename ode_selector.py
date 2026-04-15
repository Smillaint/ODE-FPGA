import os
import subprocess


def call_c_selector(client_id, round_num, buffer_size, speed):
    base_dir    = r"C:\Users\31854\ODE_FL"
    selector    = os.path.join(base_dir, "ode_selector", "selector.exe")
    grad_file   = f"grad_client{client_id}.txt"
    global_file = "global_grad.txt"

    if not os.path.exists(selector):
        print(f"  ❌ 找不到 selector.exe: {selector}")
        return []
    if not os.path.exists(os.path.join(base_dir, grad_file)):
        print(f"  ❌ 找不到梯度文件: {grad_file}")
        return []
    if not os.path.exists(os.path.join(base_dir, global_file)):
        print(f"  ❌ 找不到全局梯度文件: {global_file}")
        return []

    cmd = [selector, grad_file, global_file,
           str(round_num), str(buffer_size), str(speed)]
    print(f"  [CMD] {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=base_dir,       # ← 工作目录必须是 ODE_FL\
            timeout=60
        )
        if result.returncode != 0:
            print(f"  ❌ selector.exe 异常退出，返回码: {result.returncode}")
            print(f"  stderr: {result.stderr}")
            return []

        selected = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line.startswith("Selected:"):
                try:
                    idx = int(line.split(":")[1].strip())
                    selected.append(idx)
                except ValueError:
                    pass

        print(f"  selector 输出: {result.stdout.strip()}")
        return selected

    except subprocess.TimeoutExpired:
        print("  ❌ selector.exe 超时（超过 60 秒）")
        return []
    except Exception as e:
        print(f"  ❌ 调用 selector.exe 失败: {e}")
        return []
