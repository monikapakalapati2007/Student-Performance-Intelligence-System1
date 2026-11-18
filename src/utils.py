"""Shared utilities: reproducibility, hardware detection, logging, IO helpers."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

SEED = 1337


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and torch (if available) for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy always present in practice
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Hardware detection
# --------------------------------------------------------------------------- #
def _total_ram_gb() -> Optional[float]:
    """Best-effort total system memory in GB without extra dependencies."""
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullTotalPhys / (1024 ** 3), 2)
            return None
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return round(page * pages / (1024 ** 3), 2)
    except Exception:
        return None


def _cpu_name() -> str:
    try:
        if os.name == "nt":
            return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor() or "unknown")
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            return out.stdout.strip() or "unknown"
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def get_hardware_info() -> Dict[str, Any]:
    """Detect OS, python, torch, CPU, GPU/CUDA, MPS and RAM."""
    info: Dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": None,
        "cpu_name": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "total_ram_gb": _total_ram_gb(),
        "cuda_available": False,
        "cuda_version": None,
        "cuda_device_count": 0,
        "gpu_names": [],
        "gpu_total_vram_gb": None,
        "gpu_free_vram_gb": None,
        "mps_available": False,
        "selected_device": "cpu",
        "precision": "float32",
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if info["cuda_available"]:
            names, total, free = [], 0.0, 0.0
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                names.append(f"{props.name} ({props.total_memory / 1024 ** 3:.1f} GB)")
                total += props.total_memory / 1024 ** 3
                free_bytes, _ = torch.cuda.mem_get_info(i)
                free += free_bytes / 1024 ** 3
            info.update(
                cuda_device_count=torch.cuda.device_count(),
                gpu_names=names,
                gpu_total_vram_gb=round(total, 2),
                gpu_free_vram_gb=round(free, 2),
            )
        mps = getattr(torch.backends, "mps", None)
        info["mps_available"] = bool(mps is not None and mps.is_available())
        # Prefer CUDA, then Apple MPS, then CPU.
        if info["cuda_available"]:
            info["selected_device"] = "cuda"
            info["precision"] = "float16"
        elif info["mps_available"]:
            info["selected_device"] = "mps"
            info["precision"] = "float32"
        else:
            info["selected_device"] = "cpu"
            info["precision"] = "float32"
        info["threads"] = torch.get_num_threads()
    except Exception as exc:  # torch missing -> pure-python detection only
        info["torch_version"] = f"unavailable ({exc.__class__.__name__})"
        info["threads"] = os.cpu_count()
    return info


def describe_hardware(info: Dict[str, Any]) -> str:
    lines = [
        f"OS              : {info['os']}",
        f"Python          : {info['python_version']}  ({info['python_executable']})",
        f"PyTorch         : {info['torch_version']}",
        f"CPU             : {info['cpu_name']}  ({info['cpu_count']} logical cores)",
        f"System RAM      : {info['total_ram_gb']} GB",
        f"CUDA available  : {info['cuda_available']}  (CUDA build: {info['cuda_version']})",
        f"GPU devices     : {info['gpu_names'] or 'none'}",
        f"GPU VRAM        : total={info['gpu_total_vram_gb']} GB free={info['gpu_free_vram_gb']} GB",
        f"Apple MPS       : {info['mps_available']}",
        f"Selected device : {info['selected_device']}  (precision: {info['precision']})",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logger(name: str = "air_quality_slm", log_path: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_path:
        ensure_dir(os.path.dirname(os.path.abspath(log_path)))
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


@dataclass
class Timer:
    """Simple wall-clock timer used for reporting real durations."""

    start_time: float = field(default_factory=time.perf_counter)
    marks: Dict[str, float] = field(default_factory=dict)

    def mark(self, name: str) -> float:
        self.marks[name] = time.perf_counter() - self.start_time
        return self.marks[name]

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time


def format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m {sec:04.1f}s"


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False, default=_json_default)
        fh.write("\n")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def count_parameters(model: Any, trainable_only: bool = True) -> int:
    params = model.parameters()
    return sum(p.numel() for p in params if (p.requires_grad or not trainable_only))


def human_int(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"
