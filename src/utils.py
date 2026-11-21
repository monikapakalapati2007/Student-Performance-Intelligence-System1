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

    # A GPU can exist and still be unusable by the installed PyTorch wheel (CPU build).
    if not info["cuda_available"] and not info["gpu_names"]:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            detected = [line.strip() for line in out.stdout.splitlines() if line.strip()]
            if detected:
                info["gpu_names"] = detected
                info["gpu_detected_but_unusable"] = True
                try:
                    import re

                    mib = sum(float(re.search(r"([\d.]+)\s*MiB", d).group(1)) for d in detected)
                    info["gpu_total_vram_gb"] = round(mib / 1024, 2)
                except Exception:
                    pass
        except Exception:
            pass
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


# --------------------------------------------------------------------------- #
# Tabular pipeline helpers (student performance model)
# --------------------------------------------------------------------------- #
DATASET_ROOT_CANDIDATES = ("data", "hugging face", ".")

MILLION_SIGNATURE = {"weekly_self_study_hours", "attendance_percentage",
                     "class_participation", "total_score", "grade"}
EXAMS_SIGNATURE = {"gender", "race/ethnicity", "parental level of education",
                   "lunch", "test preparation course",
                   "math score", "reading score", "writing score"}

_SPLIT_WORDS = (("valid", "validation"), ("test", "test"), ("train", "train"))


def project_root() -> str:
    """Absolute path of the project root (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _split_from_name(path: str) -> str:
    name = os.path.basename(path).lower()
    for word, split in _SPLIT_WORDS:
        if word in name:
            return split
    return "unsplit"


def discover_datasets(root: Optional[str] = None) -> Dict[str, list]:
    """Find every table on disk and group it by dataset family and split.

    Returns ``{family: {split: [paths]}}``.  Files are only *read* here; nothing
    is moved, renamed or rewritten, so the original downloads stay untouched.
    """
    import pandas as pd

    root = root or project_root()
    skip_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__", "processed"}
    found: Dict[str, Dict[str, list]] = {"million": {}, "exams": {}}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        for fname in filenames:
            lower = fname.lower()
            if not lower.endswith((".parquet", ".csv")):
                continue
            path = os.path.join(dirpath, fname)
            try:
                if lower.endswith(".parquet"):
                    import pyarrow.parquet as pq

                    columns = set(pq.read_schema(path).names)
                else:
                    columns = set(pd.read_csv(path, nrows=0).columns)
            except Exception:
                continue
            if MILLION_SIGNATURE <= columns:
                family = "million"
            elif EXAMS_SIGNATURE <= columns:
                family = "exams"
            else:
                continue
            split = _split_from_name(path)
            found[family].setdefault(split, []).append(path)
    for family in found:
        for split in found[family]:
            found[family][split].sort()
    return found


def read_tables(paths: list, **kwargs):
    """Concatenate parquet/csv tables without modifying the source files."""
    import pandas as pd

    frames = []
    for path in paths:
        if path.lower().endswith(".parquet"):
            frames.append(pd.read_parquet(path, **kwargs))
        else:
            frames.append(pd.read_csv(path, **kwargs))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """MAE / MSE / RMSE / R2 plus the target's spread for context."""
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n_samples": int(y_true.size),
        "target_mean": float(y_true.mean()),
        "target_std": float(y_true.std()),
    }


def classification_metrics(y_true, y_pred, y_proba=None, labels=None) -> Dict[str, Any]:
    """Accuracy / precision / recall / F1 / confusion matrix / (optional) ROC-AUC."""
    import numpy as np
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_samples": int(y_true.size),
    }
    label_list = list(labels) if labels is not None else sorted(set(y_true.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=label_list)
    out["labels"] = [str(x) for x in label_list]
    out["confusion_matrix"] = cm.tolist()
    if y_proba is not None:
        try:
            proba = np.asarray(y_proba, dtype=float)
            if proba.shape[0] == y_true.size and len(label_list) > 2:
                out["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                                  labels=label_list))
            elif proba.shape[1] == 2 and len(label_list) == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
        except ValueError as exc:  # e.g. a class absent from y_true
            out["roc_auc_error"] = str(exc)
    return out
