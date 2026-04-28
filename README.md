# ODE-FPGA

联邦学习边缘加速项目的 Python 端组件，支持 ODE/CSD 样本选择流程，并可选接入 FPGA HLS 仿真选择器。

本仓库负责联邦学习实验主流程：加载流式数据、生成样本级梯度、调用样本选择器、根据选择结果进行客户端本地训练、服务器聚合全局更新，并输出与 FedDyS 基线的对比结果。配套的 HLS 组件仓库是 `csd-hls`，用于计算样本梯度与全局梯度之间的余弦相似度，并输出 Top-K 高价值样本索引。

## 项目定位

该项目面向边缘端联邦学习场景。在边缘设备算力、内存和通信资源受限的情况下，每个客户端不直接使用全部本地样本训练，而是先对当前轮的数据流窗口生成样本级梯度，再通过 ODE/CSD 选择器筛选更重要的数据样本。

Python 端主要承担以下职责：

- 构建 MNIST 流式联邦学习实验。
- 为每个客户端生成 `grad_client*.txt` 样本梯度文件。
- 在服务器聚合后生成 `global_grad.txt` 全局梯度文件。
- 调用 C 选择器，或在配置后调用 HLS 生成的 `csim.exe`。
- 读取 `selected_indices.txt`，得到被选择的样本索引。
- 使用被选择样本完成客户端本地训练。
- 保存 ODE-FL 与 FedDyS 的实验结果和对比图。

## 整体流程

```text
MNIST 流式数据
      |
      v
FederatedClient.generate_grads_for_round()
      |
      +--> grad_client0.txt / grad_client1.txt / grad_client2.txt
      |
      v
call_c_selector()
      |
      +--> 可选：HLS csim 选择器
      +--> 回退：C 选择器
      |
      v
selected_indices.txt
      |
      v
FederatedClient.train_on_selected()
      |
      v
FederatedServer.aggregate()
      |
      +--> global_grad.txt，供下一轮选择使用
```

第 0 轮还没有上一轮聚合得到的全局梯度，因此会跳过 ODE/CSD 选择，先完成一次常规训练和服务器聚合。从第 1 轮开始，选择器使用 `global_grad.txt` 中的全局梯度参与样本重要性评分。

## 目录结构

```text
main.py                 ODE-FL 主实验入口
main_feddys.py          FedDyS 基线实验入口
compare_results.py      ODE-FL 与 FedDyS 对比绘图脚本
client.py               联邦客户端逻辑：样本梯度生成、本地训练
server.py               联邦服务器逻辑：参数差聚合、全局梯度写出
model.py                SimpleCNN 模型与梯度向量工具
data_utils.py           MNIST 加载与流式数据窗口切分
grad_writer.py          梯度交换文件写入与清理
ode_selector.py         C/HLS 选择器调用封装
config.py               路径与外部选择器配置
ode_selector/           C 选择器源码和可执行文件目录
results/                实验结果 JSON 和图像输出
```

## 路径配置

代码已经移除本机硬编码路径。默认路径会根据仓库所在位置自动推导：

- 项目根目录：`config.py` 所在目录。
- 梯度交换目录：默认是项目根目录。
- C 选择器：默认是 `ode_selector/selector.exe`。
- HLS 选择器：默认不启用，需要显式配置。

如需更换目录或接入 HLS 仿真器，可以通过环境变量覆盖：

```powershell
$env:ODE_GRAD_DIR="C:\path\to\grad_exchange"
$env:ODE_C_SELECTOR="C:\path\to\selector.exe"
$env:ODE_HLS_CSIM="C:\path\to\csim.exe"
```

其中 `ODE_HLS_CSIM` 应指向 HLS 工程生成的 `csim.exe`。如果没有设置该变量，程序会直接使用 C 选择器；如果设置后调用失败，`ode_selector.py` 会自动回退到 C 选择器。

## 环境依赖

建议使用 Python 3.9 或更高版本。

```bash
pip install torch torchvision numpy matplotlib
```

MNIST 数据集会在首次运行时自动下载到 `data/` 目录。

## 运行方式

运行 ODE-FL 实验：

```bash
python main.py
```

运行 FedDyS 基线实验：

```bash
python main_feddys.py
```

生成对比图：

```bash
python compare_results.py
```

运行完成后，结果文件保存在：

```text
results/baseline_ode.json
results/baseline_ode.png
results/baseline_feddys.json
results/baseline_feddys.png
results/comparison.png
```

## 与 HLS 组件联动

Python 端会为选择器生成输入文件：

```text
grad_client0.txt
grad_client1.txt
grad_client2.txt
global_grad.txt
```

HLS 或 C 选择器读取客户端梯度和全局梯度后，输出：

```text
selected_indices.txt
```

`ode_selector.py` 会解析该文件并返回全局样本索引。随后 `client.py` 将全局样本索引映射回当前数据流窗口中的局部索引，用于本轮本地训练。

## 主要实验参数

主要参数定义在 `main.py`：

```python
NUM_CLIENTS = 3
NUM_ROUNDS = 50
SPEED = 100
BUFFER_SIZE = 20
LR = 0.05
LOCAL_EPOCHS = 5
```

`SPEED` 表示每轮每个客户端处理的数据流窗口大小，`BUFFER_SIZE` 表示选择器每轮保留的样本数量。

## 运行时生成文件

以下文件属于运行时交换文件或缓存文件，不应提交到 Git：

```text
grad_client*.txt
global_grad.txt
selected_indices.txt
__pycache__/
data/
```

这些文件已在 `.gitignore` 中排除。`results/` 目录中的基准结果保留在仓库中，便于展示实验输出。

## 参考项目

- ODE: https://github.com/gongchenooo/WWW23-ODE
- FedDyS: https://github.com/ensiyeKiya/FedDyS
