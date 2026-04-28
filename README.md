# ODE-FPGA

Python-side federated learning pipeline for ODE/CSD-based edge acceleration, with optional FPGA HLS selector integration.

This repository contains the Python-side component of a federated learning edge-acceleration project. It runs the FL experiment loop, writes gradient exchange files, calls a sample selector, trains clients on selected samples, aggregates client updates, and produces comparison results against the FedDyS baseline.

The companion HLS repository is `csd-hls`. That component computes cosine-similarity importance scores between sample gradients and the global gradient, then emits Top-K sample indices for the Python training loop.

## Role In The System

The project targets edge-side federated learning under limited compute, memory, and communication budget. Each client receives a streaming data window each round. The Python pipeline extracts per-sample gradients, invokes a selector, keeps high-value samples, performs local training, and sends the resulting parameter delta to the server.

At a high level, this repository handles:

- MNIST streaming federated learning experiments.
- Per-client sample-gradient file generation.
- Global-gradient file generation after server aggregation.
- C selector invocation with optional HLS `csim.exe` acceleration.
- Selected-index parsing from `selected_indices.txt`.
- ODE-FL and FedDyS result generation and plotting.

## Pipeline

```text
MNIST stream data
      |
      v
FederatedClient.generate_grads_for_round()
      |
      +--> grad_client0.txt / grad_client1.txt / grad_client2.txt
      |
      v
call_c_selector()
      |
      +--> optional HLS csim selector
      +--> fallback C selector
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
      +--> global_grad.txt for the next round
```

Round 0 has no previous global gradient, so selector-based pruning is skipped for that first round. Starting from round 1, the selector uses the previous aggregation result stored in `global_grad.txt`.

## Repository Layout

```text
main.py                 ODE-FL experiment entry point
main_feddys.py          FedDyS baseline entry point
compare_results.py      ODE-FL vs FedDyS plotting script
client.py               Federated client logic and local training
server.py               Server aggregation and global-gradient output
model.py                SimpleCNN model and gradient-vector helper
data_utils.py           MNIST loading and streaming-window slicing
grad_writer.py          Gradient exchange-file writer and cleanup helper
ode_selector.py         C/HLS selector invocation wrapper
config.py               Path and external selector configuration
ode_selector/           C selector source/executable location
results/                Experiment JSON and figure outputs
```

## Path Configuration

Hard-coded local paths have been removed. Defaults are derived from the repository location:

- Project root: directory containing `config.py`.
- Gradient exchange directory: project root.
- C selector: `ode_selector/selector.exe`.
- HLS selector: disabled by default unless configured.

Override paths with environment variables when needed:

```powershell
$env:ODE_GRAD_DIR="C:\path	o\grad_exchange"
$env:ODE_C_SELECTOR="C:\path	o\selector.exe"
$env:ODE_HLS_CSIM="C:\path	o\csim.exe"
```

`ODE_HLS_CSIM` should point to the HLS-generated `csim.exe` from the companion HLS project. If it is unset, the Python pipeline uses the C selector directly. If it is set but fails at runtime, `ode_selector.py` falls back to the C selector.

## Requirements

Python 3.9+ is recommended.

```bash
pip install torch torchvision numpy matplotlib
```

MNIST is downloaded automatically into `data/` on first run.

## Usage

Run the ODE-FL experiment:

```bash
python main.py
```

Run the FedDyS baseline:

```bash
python main_feddys.py
```

Generate the comparison plot:

```bash
python compare_results.py
```

Expected result files:

```text
results/baseline_ode.json
results/baseline_ode.png
results/baseline_feddys.json
results/baseline_feddys.png
results/comparison.png
```

## HLS Integration

The Python side writes selector inputs:

```text
grad_client0.txt
grad_client1.txt
grad_client2.txt
global_grad.txt
```

The selector writes:

```text
selected_indices.txt
```

`ode_selector.py` parses that file and returns global sample indices. `client.py` maps those global indices back into the current streaming window before local training.

## Main Experiment Parameters

The primary settings are in `main.py`:

```python
NUM_CLIENTS = 3
NUM_ROUNDS = 50
SPEED = 100
BUFFER_SIZE = 20
LR = 0.05
LOCAL_EPOCHS = 5
```

`SPEED` is the per-client streaming window size per round. `BUFFER_SIZE` is the number of samples retained by the selector.

## Generated Files

These runtime files are intentionally ignored by Git:

```text
grad_client*.txt
global_grad.txt
selected_indices.txt
__pycache__/
data/
```

The `results/` directory is kept in the repository so baseline outputs can be shown with the code.

## References

- ODE: https://github.com/gongchenooo/WWW23-ODE
- FedDyS: https://github.com/ensiyeKiya/FedDyS
