"""Dataset and environment inspection for the student-performance project.

    python src/inspect_data.py

Reads every table found on disk (read-only), reports shape, dtypes, missing
values, duplicates, outliers, target behaviour and cross-split overlap, plus the
hardware/software environment.  Writes:

    reports/data_inspection.md
    results/data_inspection.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import (discover_datasets, get_hardware_info, project_root,
                   read_tables, save_json)  # noqa: E402

MILLION_TARGET = "total_score"
MILLION_FEATURES = ["weekly_self_study_hours", "attendance_percentage", "class_participation"]
EXAMS_FEATURES = ["gender", "race/ethnicity", "parental level of education", "lunch",
                  "test preparation course"]
EXAMS_TARGETS = ["math score", "reading score", "writing score"]
GRADE_BANDS = [("A", 85.0), ("B", 70.0), ("C", 55.0), ("D", 40.0), ("F", -np.inf)]


def _outlier_summary(series: pd.Series) -> Dict[str, float]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    return {
        "min": float(series.min()), "q1": float(q1), "median": float(series.median()),
        "q3": float(q3), "max": float(series.max()),
        "mean": float(series.mean()), "std": float(series.std()),
        "iqr_3x_low": float(lo), "iqr_3x_high": float(hi),
        "n_outside_3iqr": int(((series < lo) | (series > hi)).sum()),
        "n_zero": int((series == 0).sum()),
    }


def _grade_from_score(score: float) -> str:
    for label, cutoff in GRADE_BANDS:
        if score >= cutoff:
            return label
    return "F"


def _overlap(a: pd.DataFrame, b: pd.DataFrame, keys: List[str]) -> int:
    if not keys:
        return 0
    left = set(map(tuple, a[keys].astype(str).to_numpy()))
    right = set(map(tuple, b[keys].astype(str).to_numpy()))
    return len(left & right)


def inspect_million(frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    info: Dict[str, Any] = {"family": "million", "splits": {}}
    all_rows = []
    for split, df in frames.items():
        all_rows.append(df)
        dup_mask = df.duplicated()
        info["splits"][split] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "missing": {c: int(v) for c, v in df.isna().sum().items()},
            "duplicate_rows": int(dup_mask.sum()),
            "unique_student_ids": int(df["student_id"].nunique()),
        }
    full = pd.concat(all_rows, ignore_index=True)
    info["rows_total"] = int(len(full))
    info["dtypes"] = {c: str(t) for c, t in full.dtypes.items()}
    info["numeric_summary"] = {
        c: _outlier_summary(full[c].astype(float))
        for c in MILLION_FEATURES + [MILLION_TARGET]
    }
    info["target"] = MILLION_TARGET
    info["grade_counts"] = {str(k): int(v) for k, v in full["grade"].value_counts().items()}
    info["grade_vs_target"] = {
        str(g): {"min": float(s.min()), "max": float(s.max()), "count": int(len(s))}
        for g, s in full.groupby("grade")[MILLION_TARGET]
    }
    derived = full[MILLION_TARGET].map(_grade_from_score)
    info["grade_is_deterministic_function_of_target"] = bool((derived == full["grade"]).all())
    info["grade_mismatch_rows"] = int((derived != full["grade"]).sum())
    info["grade_agreement_rate"] = float((derived == full["grade"]).mean())
    boundary = full[derived != full["grade"]][MILLION_TARGET]
    info["grade_mismatch_at_cutoffs_only"] = bool(boundary.isin([40.0, 55.0, 70.0, 85.0]).all())
    info["target_at_ceiling_rows"] = int((full[MILLION_TARGET] == full[MILLION_TARGET].max()).sum())
    info["correlation_with_target"] = {
        c: float(full[c].astype(float).corr(full[MILLION_TARGET].astype(float)))
        for c in MILLION_FEATURES
    }
    info["student_id_correlation_with_target"] = float(
        full["student_id"].astype(float).corr(full[MILLION_TARGET].astype(float))
    )
    info["duplicate_rows_across_splits"] = _overlap(
        frames.get("train", full.iloc[0:0]), frames.get("test", full.iloc[0:0]), ["student_id"]
    )
    info["train_test_student_id_overlap"] = _overlap(
        frames.get("train", full.iloc[0:0]), frames.get("test", full.iloc[0:0]), ["student_id"]
    )
    info["train_validation_student_id_overlap"] = _overlap(
        frames.get("train", full.iloc[0:0]), frames.get("validation", full.iloc[0:0]), ["student_id"]
    )
    return info


def inspect_exams(frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    info: Dict[str, Any] = {"family": "exams", "splits": {}}
    for split, df in frames.items():
        info["splits"][split] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "missing": {c: int(v) for c, v in df.isna().sum().items()},
            "duplicate_rows": int(df.duplicated().sum()),
        }
    full = pd.concat(frames.values(), ignore_index=True)
    info["rows_total"] = int(len(full))
    info["dtypes"] = {c: str(t) for c, t in full.dtypes.items()}
    info["categorical_levels"] = {c: {str(k): int(v) for k, v in full[c].value_counts().items()}
                                  for c in EXAMS_FEATURES}
    info["numeric_summary"] = {c: _outlier_summary(full[c].astype(float)) for c in EXAMS_TARGETS}
    info["correlation"] = {c: float(full[c].astype(float).corr(full["math score"].astype(float)))
                           for c in EXAMS_TARGETS}
    info["train_test_feature_row_overlap"] = _overlap(
        frames.get("train", full.iloc[0:0]), frames.get("test", full.iloc[0:0]), EXAMS_FEATURES
    )
    info["train_validation_feature_row_overlap"] = _overlap(
        frames.get("train", full.iloc[0:0]), frames.get("validation", full.iloc[0:0]), EXAMS_FEATURES
    )
    return info


def environment() -> Dict[str, Any]:
    hw = get_hardware_info()
    versions: Dict[str, Any] = {}
    for name in ("numpy", "pandas", "sklearn", "joblib", "matplotlib", "pyarrow"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover
            versions[name] = f"missing ({exc.__class__.__name__})"
    hw["package_versions"] = versions
    try:
        import torch
        hw["torch_cuda_build"] = getattr(torch.version, "cuda", None)
        hw["torch_note"] = ("torch CPU build; the installed RTX GPU is not usable by this "
                            "PyTorch wheel" if hw["gpu_names"] and not hw["cuda_available"]
                            else "")
    except Exception:
        pass
    return hw


def build_report() -> Dict[str, Any]:
    started = time.time()
    found = discover_datasets(project_root())
    report: Dict[str, Any] = {
        "discovered_files": {fam: {s: [os.path.relpath(p, project_root()) for p in paths]
                                   for s, paths in splits.items()}
                             for fam, splits in found.items()},
        "environment": environment(),
    }

    million_frames = {s: read_tables(paths) for s, paths in found["million"].items()}
    exams_frames = {s: read_tables(paths) for s, paths in found["exams"].items()}
    if million_frames:
        report["million"] = inspect_million(million_frames)
    if exams_frames:
        report["exams"] = inspect_exams(exams_frames)

    compat: List[str] = []
    m_cols = set(million_frames.get("train", pd.DataFrame()).columns)
    e_cols = set(exams_frames.get("train", pd.DataFrame()).columns)
    if m_cols and e_cols:
        compat.append(f"shared columns: {sorted(m_cols & e_cols) or 'none'}")
        compat.append("feature sets are disjoint (study behaviour/demographics); targets are on "
                      "different scales (single bounded total_score vs three exam scores)")
        compat.append("no row-level key links the two datasets, and concatenating them would "
                      "create rows with 5-6 missing features")
        compat.append("DECISION: datasets are incompatible for a single joint model -> trained "
                      "as two separate models, with the 1M-record dataset as primary")
    report["compatibility"] = compat
    report["inspection_seconds"] = round(time.time() - started, 2)
    return report


def render_markdown(report: Dict[str, Any]) -> str:
    env = report["environment"]
    lines = ["# Dataset & environment inspection", "",
             "Generated by `python src/inspect_data.py`. All datasets are opened read-only.", "",
             "## Environment", "",
             f"- OS: {env['os']}",
             f"- Python: {env['python_version']} (`{env['python_executable']}`)",
             f"- CPU: {env['cpu_name']} — {env['cpu_count']} logical cores",
             f"- System RAM: {env['total_ram_gb']} GB",
             f"- GPUs detected: {env['gpu_names'] or 'none'} "
             f"(VRAM total {env['gpu_total_vram_gb']} GB)",
             f"- CUDA available to PyTorch: {env['cuda_available']} "
             f"(torch {env['torch_version']}, CUDA build {env.get('torch_cuda_build')})",
             f"- Selected compute device: {env['selected_device']}",
             f"- Package versions: {json.dumps(env['package_versions'])}", ""]
    if env.get("torch_note"):
        lines.append(f"> {env['torch_note']}")
        lines.append("")
    lines += ["## Discovered files", ""]
    for fam, splits in report["discovered_files"].items():
        if not splits:
            continue
        lines.append(f"**{fam}**")
        for split, paths in sorted(splits.items()):
            for p in paths:
                lines.append(f"- {split}: `{p}`")
        lines.append("")
    if "million" in report:
        m = report["million"]
        lines += ["## Dataset 1 — Student Performance (1M records)", "",
                  f"Total rows: **{m['rows_total']:,}** — target `{m['target']}`", "",
                  "| split | rows | duplicates | missing |", "|---|---|---|---|"]
        for split, s in m["splits"].items():
            miss = sum(s["missing"].values())
            lines.append(f"| {split} | {s['rows']:,} | {s['duplicate_rows']:,} | {miss} |")
        lines += ["", "### Numeric summary (3×IQR used for outlier counting)", "",
                  "| column | min | median | max | mean | std | outliers | zeros |",
                  "|---|---|---|---|---|---|---|---|"]
        for col, s in m["numeric_summary"].items():
            lines.append(f"| {col} | {s['min']:.2f} | {s['median']:.2f} | {s['max']:.2f} | "
                         f"{s['mean']:.2f} | {s['std']:.2f} | {s['n_outside_3iqr']} | "
                         f"{s['n_zero']} |")
        lines += ["", "### Target behaviour", "",
                  f"- Grade counts: {json.dumps(m['grade_counts'])}",
                  f"- Grade agreement with the documented cut-offs (A>=85, B>=70, C>=55, D>=40): "
                  f"**{m['grade_agreement_rate'] * 100:.2f} %** "
                  f"({m['grade_mismatch_rows']} mismatching rows, all at exact cut-off values: "
                  f"{m['grade_mismatch_at_cutoffs_only']}) -> `grade` is a near-deterministic "
                  f"binning of the score, so it is never used as a grade-model feature",
                  f"- Rows at the 100-point ceiling: **{m['target_at_ceiling_rows']:,}** "
                  f"({m['target_at_ceiling_rows'] / max(m['rows_total'], 1) * 100:.1f} %) — the "
                  f"target is right-censored, which caps achievable R².",
                  f"- Correlation with target: {json.dumps({k: round(v, 3) for k, v in m['correlation_with_target'].items()})}",
                  f"- `student_id` correlation with target: "
                  f"{m['student_id_correlation_with_target']:.4f} (an ID, excluded from features)",
                  f"- Student-ID overlap train/test: {m['train_test_student_id_overlap']}, "
                  f"train/validation: {m['train_validation_student_id_overlap']}",
                  ""]
    if "exams" in report:
        e = report["exams"]
        lines += ["## Dataset 2 — Students Performance in Exams", "",
                  f"Total rows: **{e['rows_total']:,}**, columns: "
                  f"{', '.join(f'`{c}`' for c in e['dtypes'])}", "",
                  "| split | rows | duplicates |", "|---|---|---|"]
        for split, s in e["splits"].items():
            lines.append(f"| {split} | {s['rows']:,} | {s['duplicate_rows']} |")
        lines += ["", "| column | levels |", "|---|---|"]
        for c, levels in e["categorical_levels"].items():
            lines.append(f"| {c} | {json.dumps(levels)} |")
        lines += ["", "| score | min | median | max | std | outliers |", "|---|---|---|---|---|---|"]
        for c, s in e["numeric_summary"].items():
            lines.append(f"| {c} | {s['min']:.1f} | {s['median']:.1f} | {s['max']:.1f} | "
                         f"{s['std']:.2f} | {s['n_outside_3iqr']} |")
        lines += ["", f"- Correlation of reading/writing with math: "
                  f"{json.dumps({k: round(v, 3) for k, v in e['correlation'].items()})}",
                  f"- Feature-row overlap train/test: {e['train_test_feature_row_overlap']}, "
                  f"train/validation: {e['train_validation_feature_row_overlap']}", ""]
    lines += ["## Dataset compatibility", ""] + [f"- {c}" for c in report["compatibility"]] + [""]
    return "\n".join(lines)


def write_manifest(report: Dict[str, Any]) -> Dict[str, Any]:
    """Record where the original files live and what they contain (no copying)."""
    manifest: Dict[str, Any] = {"note": "original files are read-only; processed copies live in "
                                      "data/processed/tabular/", "datasets": {}}
    found = discover_datasets(project_root())
    for family, splits in found.items():
        entry: Dict[str, Any] = {"files": {}}
        for split, paths in splits.items():
            for path in paths:
                df_head = read_tables([path]).head(5)
                entry["files"][os.path.relpath(path, project_root())] = {
                    "split": split, "columns": list(df_head.columns),
                    "dtypes": {c: str(t) for c, t in df_head.dtypes.items()},
                }
        if family in report:
            entry["rows_total"] = report[family]["rows_total"]
        manifest["datasets"][family] = entry
    path = os.path.join(PROJECT_ROOT, "data", "raw", "dataset_manifest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_json(manifest, path)
    return manifest


def main() -> int:
    report = build_report()
    save_json(report, os.path.join(PROJECT_ROOT, "results", "data_inspection.json"))
    write_manifest(report)
    md = render_markdown(report)
    path = os.path.join(PROJECT_ROOT, "reports", "data_inspection.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
