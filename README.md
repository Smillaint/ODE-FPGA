# ODE-FL: Federated Learning with ODE-based Data Selector

## 项目简介

本项目提出了一种基于 ODE（常微分方程）的数据选择器，应用于联邦学习框架中，
旨在提升客户端本地训练的数据质量，并与 FedDyS 基线方法进行对比实验。

---ODE,FedDyS方法项目地址
https://github.com/gongchenooo/WWW23-ODE
https://github.com/ensiyeKiya/FedDyS

## 环境依赖

- Python 3.9+
- PyTorch
- NumPy
- Matplotlib

安装依赖：
```bash
pip install torch numpy matplotlib
数据集
使用 MNIST 数据集，首次运行时会自动下载到 data/ 目录，无需手动准备。

运行方式
# 运行 ODE-FL 主程序
python main.py

# 运行 FedDyS 基线对比
python main_feddys.py

# 生成对比结果图
python compare_results.py
项目结构
ODE_FL/
├── main.py                # ODE-FL 主程序
├── main_feddys.py         # FedDyS 基线程序
├── compare_results.py     # 结果对比可视化
├── ode_selector/          # ODE 数据选择器模块（含 C 扩展）
├── data/                  # 数据集目录（自动生成）
└── results/               # 实验结果图片输出
实验结果
运行 compare_results.py 后，结果图片将保存在 results/ 目录下。

作者
GitHub: @Smillaint