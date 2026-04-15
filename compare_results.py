import json
import matplotlib.pyplot as plt

# 加载两个实验结果
with open("results/baseline_ode.json")    as f: ode    = json.load(f)
with open("results/baseline_feddys.json") as f: feddys = json.load(f)

rounds_ode    = [r['round']    for r in ode]
accs_ode      = [r['accuracy'] for r in ode]
rounds_feddys = [r['round']    for r in feddys]
accs_feddys   = [r['accuracy'] for r in feddys]

plt.figure(figsize=(10, 6))
plt.plot(rounds_ode,    accs_ode,    'b-o', label='ODE',    linewidth=2)
plt.plot(rounds_feddys, accs_feddys, 'r-s', label='FedDyS', linewidth=2)
plt.xlabel('Communication Round', fontsize=13)
plt.ylabel('Test Accuracy (%)',   fontsize=13)
plt.title('ODE vs FedDyS on MNIST (20% data, streaming)',  fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("results/comparison.png", dpi=150)
plt.show()
print("✅ 对比图已保存为 results/comparison.png")
