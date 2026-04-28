import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _env_path(name, default):
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else Path(default).resolve()


GRAD_DIR = _env_path("ODE_GRAD_DIR", PROJECT_ROOT)
C_SELECTOR_EXE = _env_path(
    "ODE_C_SELECTOR",
    PROJECT_ROOT / "ode_selector" / "selector.exe",
)
HLS_CSIM_EXE = _env_path("ODE_HLS_CSIM", "")


def grad_path(filename):
    return GRAD_DIR / filename


def ensure_grad_dir():
    GRAD_DIR.mkdir(parents=True, exist_ok=True)
