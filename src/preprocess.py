"""Preprocessing for the student-performance models.

    python src/preprocess.py            # build data/processed/tabular/* and data/splits/*
    python src/preprocess.py --report   # only print what is on disk

Design decisions (measured in reports/data_inspection.md):

* Dataset 1 (1M records) — primary.  Features are study behaviour only
  (weekly study hours, attendance %, class participation); ``total_score`` is the
  regression target and ``grade`` the classification target.  ``student_id`` is
  dropped (an identifier, not a behaviour).  ``grade`` is an exact binning of
  ``total_score``, so grade models never receive the score as an input.
* Dataset 2 (1k records) — secondary.  Five categorical features; ``math score``
  is the regression target.  Trained separately because the schemas are not
  compatible.
* The supplied train/validation/test files are used as the splits; disjointness
  is *verified*, and the data are re-split only if the provided splits overlap.

Fitted objects (scaler, encoders, feature order) are saved with joblib so that
inference cannot drift from training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import discover_datasets, read_tables, save_json, set_seed  # noqa: E402

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "tabular")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
PREPROCESSOR_DIR = os.path.join(PROJECT_ROOT, "models", "student_performance", "preprocessor")

RAW_FEATURES = ["weekly_self_study_hours", "attendance_percentage", "class_participation"]
TARGET_COLUMN = "total_score"
GRADE_COLUMN = "grade"
STUDY_CAP_HOURS = 40.0
GRADE_CUTOFFS = {"A": 85.0, "B": 70.0, "C": 55.0, "D": 40.0}  # verified against the data

EXAM_FEATURES = ["gender", "race/ethnicity", "parental level of education", "lunch",
                 "test preparation course"]
EXAM_TARGET = "math score"
# Explicit, documented thresholds for the optional exams performance categories.
EXAM_CATEGORY_CUTOFFS = {"needs improvement": 60.0, "satisfactory": 80.0}  # <60 / 60-79 / 80+
EXAM_CATEGORIES = ["needs improvement", "satisfactory", "good"]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, row-local features (no statistic learned from other rows)."""
    out = df.copy()
    hours = out["weekly_self_study_hours"].clip(0, STUDY_CAP_HOURS)
    attendance = out["attendance_percentage"].clip(0, 100)
    participation = out["class_participation"].clip(0, 10)
    out["effort_index"] = hours * participation / 10.0
    out["study_attendance_product"] = hours * attendance / 100.0
    out["participation_per_study_hour"] = participation / (hours + 1.0)
    return out


ENGINEERED_FEATURES = ["effort_index", "study_attendance_product", "participation_per_study_hour"]
MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


def grade_from_score(score: float) -> str:
    for label, cutoff in GRADE_CUTOFFS.items():
        if score >= cutoff:
            return label
    return "F"


def exam_category(score: float) -> str:
    if score < EXAM_CATEGORY_CUTOFFS["needs improvement"]:
        return "needs improvement"
    if score < EXAM_CATEGORY_CUTOFFS["satisfactory"]:
        return "satisfactory"
    return "good"


# ---------------------------------------------------------------------------
# Dataset 1 — 1M student records
# ---------------------------------------------------------------------------
def load_million_splits(verbose: bool = True) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    found = discover_datasets(PROJECT_ROOT)["million"]
    missing = [s for s in ("train", "test") if not found.get(s)]
    if missing:
        raise FileNotFoundError(f"dataset 1 is missing required splits: {missing}; found {found}")
    frames = {split: read_tables(paths) for split, paths in found.items()}
    log: Dict[str, Any] = {"source_files": {s: [os.path.basename(p) for p in paths]
                                            for s, paths in found.items()}, "splits": {}}

    cleaned: Dict[str, pd.DataFrame] = {}
    for split, df in frames.items():
        n0 = len(df)
        df = df.copy()
        for col in RAW_FEATURES + [TARGET_COLUMN]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        missing_rows = int(df[RAW_FEATURES + [TARGET_COLUMN]].isna().any(axis=1).sum())
        df = df.dropna(subset=RAW_FEATURES + [TARGET_COLUMN])
        dupes = int(df.duplicated().sum())
        df = df.drop_duplicates().reset_index(drop=True)
        # Clamp only physically impossible values; outliers are kept (reported, not removed).
        df["weekly_self_study_hours"] = df["weekly_self_study_hours"].clip(0, STUDY_CAP_HOURS)
        df["attendance_percentage"] = df["attendance_percentage"].clip(0, 100)
        df["class_participation"] = df["class_participation"].clip(0, 10)
        df = engineer_features(df)
        # Guard: the target must never be an input.
        assert TARGET_COLUMN not in MODEL_FEATURES and GRADE_COLUMN not in MODEL_FEATURES
        cleaned[split] = df
        log["splits"][split] = {
            "rows_in": int(n0), "missing_rows_dropped": missing_rows,
            "duplicate_rows_dropped": dupes, "rows_out": int(len(df)),
        }

    id_overlap = set(cleaned["train"]["student_id"]) & set(cleaned.get("test", cleaned["train"])["student_id"])
    train_val_overlap = set(cleaned["train"]["student_id"]) & set(
        cleaned.get("validation", cleaned["train"].iloc[0:0])["student_id"])
    log["train_test_student_id_overlap"] = len(id_overlap)
    log["train_validation_student_id_overlap"] = len(train_val_overlap)
    if id_overlap or train_val_overlap:
        # Re-split on the combined table so no student is seen in two splits.
        all_rows = pd.concat(cleaned.values(), ignore_index=True).drop_duplicates("student_id")
        train_df, rest = train_test_split(all_rows, test_size=0.30, random_state=1337)
        val_df, test_df = train_test_split(rest, test_size=0.50, random_state=1337)
        cleaned = {"train": train_df.reset_index(drop=True),
                   "validation": val_df.reset_index(drop=True),
                   "test": test_df.reset_index(drop=True)}
        log["resplit_due_to_overlap"] = True

    for split in ("train", "validation", "test"):
        if not cleaned.get(split, pd.DataFrame()).empty:
            log["splits"].setdefault(split, {})["rows_final"] = int(len(cleaned[split]))
    if verbose:
        for split, info in log["splits"].items():
            print(f"  {split:<11} {info.get('rows_final', info['rows_out']):>8,} rows")
    return cleaned, log


# ---------------------------------------------------------------------------
# Dataset 2 — exams
# ---------------------------------------------------------------------------
def load_exam_splits() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    found = discover_datasets(PROJECT_ROOT)["exams"]
    frames = {split: read_tables(paths) for split, paths in found.items()}
    if frames.get("train") is None:
        raise FileNotFoundError("dataset 2 (exams) has no train split")
    log: Dict[str, Any] = {"source_files": {s: [os.path.basename(p) for p in paths]
                                            for s, paths in found.items()}, "splits": {}}
    cleaned = {}
    for split, df in frames.items():
        df = df.copy().drop_duplicates().reset_index(drop=True)
        for col in EXAM_FEATURES:
            df[col] = df[col].astype(str).str.strip()
        log["splits"][split] = {"rows": int(len(df))}
        cleaned[split] = df
    overlap = len(set(map(tuple, cleaned["train"][EXAM_FEATURES].to_numpy())) &
                  set(map(tuple, cleaned.get("test", cleaned["train"].iloc[0:0])[EXAM_FEATURES].to_numpy())))
    log["train_test_feature_row_overlap"] = overlap
    # The feature grid has only 5 x 5 x 6 x 2 x 2 = 600 cells, so identical feature rows
    # appear across splits by construction. There is no student key to deduplicate on and
    # no target information is shared, so the provided splits are kept as-is.
    log["overlap_note"] = ("identical feature rows across splits are expected on a 600-cell "
                           "categorical grid; they carry no target information, so no re-split")
    return cleaned, log


# ---------------------------------------------------------------------------
# Fitted transforms
# ---------------------------------------------------------------------------
def build_million_preprocessor(train_df: pd.DataFrame) -> Dict[str, Any]:
    scaler = StandardScaler().fit(train_df[MODEL_FEATURES].to_numpy(dtype=np.float64))
    grade_order = ["F", "D", "C", "B", "A"]
    present = [g for g in grade_order if g in set(train_df[GRADE_COLUMN])]
    grade_to_index = {g: i for i, g in enumerate(present)}
    return {
        "dataset": "million",
        "feature_names": MODEL_FEATURES,
        "raw_features": RAW_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "target_column": TARGET_COLUMN,
        "grade_column": GRADE_COLUMN,
        "grade_labels": present,
        "grade_to_index": grade_to_index,
        "grade_cutoffs": GRADE_CUTOFFS,
        "scaler": scaler,
        "study_cap_hours": STUDY_CAP_HOURS,
        "n_features": len(MODEL_FEATURES),
    }


def build_exam_preprocessor(train_df: pd.DataFrame) -> Dict[str, Any]:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[EXAM_FEATURES])
    feature_names = list(encoder.get_feature_names_out(EXAM_FEATURES))
    return {
        "dataset": "exams",
        "categorical_features": EXAM_FEATURES,
        "feature_names": feature_names,
        "target_column": EXAM_TARGET,
        "encoder": encoder,
        "category_cutoffs": EXAM_CATEGORY_CUTOFFS,
        "categories": EXAM_CATEGORIES,
        "n_features": len(feature_names),
    }


def transform_million(pre: Dict[str, Any], df: pd.DataFrame, scaled: bool = True) -> np.ndarray:
    X = df[MODEL_FEATURES].to_numpy(dtype=np.float64)
    return pre["scaler"].transform(X) if scaled else X


def transform_exams(pre: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    return pre["encoder"].transform(df[EXAM_FEATURES])


def save_preprocessors(million_pre: Dict[str, Any], exam_pre: Dict[str, Any]) -> None:
    os.makedirs(PREPROCESSOR_DIR, exist_ok=True)
    joblib.dump(million_pre, os.path.join(PREPROCESSOR_DIR, "million_preprocessor.joblib"))
    joblib.dump(exam_pre, os.path.join(PREPROCESSOR_DIR, "exams_preprocessor.joblib"))
    meta = {
        "million": {k: v for k, v in million_pre.items() if k not in ("scaler",)},
        "exams": {k: v for k, v in exam_pre.items() if k not in ("encoder",)},
    }
    save_json(meta, os.path.join(PREPROCESSOR_DIR, "preprocessor_meta.json"))
    save_json({"million_features": MODEL_FEATURES, "million_target": TARGET_COLUMN,
               "million_grade_column": GRADE_COLUMN,
               "exams_features": EXAM_FEATURES, "exams_target": EXAM_TARGET,
               "exams_feature_names": exam_pre["feature_names"]},
              os.path.join(PREPROCESSOR_DIR, "feature_names.json"))


def save_single_preprocessor(pre: Dict[str, Any]) -> str:
    """Save one fitted preprocessor plus human-readable metadata and feature order."""
    os.makedirs(PREPROCESSOR_DIR, exist_ok=True)
    name = "million_preprocessor.joblib" if pre["dataset"] == "million" else "exams_preprocessor.joblib"
    path = os.path.join(PREPROCESSOR_DIR, name)
    joblib.dump(pre, path)

    meta_path = os.path.join(PREPROCESSOR_DIR, "preprocessor_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}
    meta[pre["dataset"]] = {k: v for k, v in pre.items() if k not in ("scaler", "encoder")}
    save_json(meta, meta_path)

    names_path = os.path.join(PREPROCESSOR_DIR, "feature_names.json")
    names = {}
    if os.path.exists(names_path):
        try:
            with open(names_path, encoding="utf-8") as fh:
                names = json.load(fh)
        except Exception:
            names = {}
    if pre["dataset"] == "million":
        names.update({"million_features": MODEL_FEATURES, "million_target": TARGET_COLUMN,
                      "million_grade_column": GRADE_COLUMN})
    else:
        names.update({"exams_features": EXAM_FEATURES, "exams_target": EXAM_TARGET,
                      "exams_feature_names": pre["feature_names"]})
    save_json(names, names_path)
    return path


def load_preprocessors() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    million = joblib.load(os.path.join(PREPROCESSOR_DIR, "million_preprocessor.joblib"))
    exams = joblib.load(os.path.join(PREPROCESSOR_DIR, "exams_preprocessor.joblib"))
    return million, exams


def write_artifacts(splits: Dict[str, pd.DataFrame], prefix: str) -> Dict[str, str]:
    """Persist processed splits and split manifests (feature rows only, sources untouched)."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    paths = {}
    for split, df in splits.items():
        out = os.path.join(PROCESSED_DIR, f"{prefix}_{split}.parquet")
        df.to_parquet(out, index=False)
        paths[split] = out
        cols = ["student_id"] if "student_id" in df.columns else list(df.columns)
        df[cols].to_csv(os.path.join(SPLITS_DIR, f"{prefix}_{split}.csv"), index=False)
    return paths


def run_preprocessing(verbose: bool = True) -> Dict[str, Any]:
    set_seed(1337)
    summary: Dict[str, Any] = {"seed": 1337, "features_million": MODEL_FEATURES,
                               "leakage_guards": [
                                   "student_id excluded from features",
                                   "total_score excluded from grade-classifier features",
                                   "scaler/encoder fitted on the training split only"]}
    if verbose:
        print("Dataset 1 (1M records):")
    million, m_log = load_million_splits(verbose=verbose)
    summary["million"] = m_log
    million_pre = build_million_preprocessor(million["train"])
    summary["million"]["n_features"] = million_pre["n_features"]
    summary["million"]["grade_labels"] = million_pre["grade_labels"]
    write_artifacts(million, "million")

    if verbose:
        print("Dataset 2 (exams):")
    exams, e_log = load_exam_splits()
    summary["exams"] = e_log
    exam_pre = build_exam_preprocessor(exams["train"])
    summary["exams"]["n_features"] = exam_pre["n_features"]
    write_artifacts(exams, "exams")
    if verbose:
        for split, info in e_log["splits"].items():
            print(f"  {split:<11} {info['rows']:>8,} rows")

    save_preprocessors(million_pre, exam_pre)
    summary["outputs"] = {"processed_dir": os.path.relpath(PROCESSED_DIR, PROJECT_ROOT),
                          "splits_dir": os.path.relpath(SPLITS_DIR, PROJECT_ROOT),
                          "preprocessor_dir": os.path.relpath(PREPROCESSOR_DIR, PROJECT_ROOT)}
    save_json(summary, os.path.join(PROJECT_ROOT, "results", "preprocessing_summary.json"))
    return summary


def load_processed(prefix: str = "million") -> Dict[str, pd.DataFrame]:
    """Load processed splits, rebuilding them if they are absent."""
    paths = {s: os.path.join(PROCESSED_DIR, f"{prefix}_{s}.parquet")
             for s in ("train", "validation", "test")}
    if not all(os.path.exists(p) for p in paths.values()):
        run_preprocessing(verbose=False)
    return {s: pd.read_parquet(p) for s, p in paths.items() if os.path.exists(p)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build processed student-performance splits.")
    parser.add_argument("--report", action="store_true", help="print the saved preprocessing summary")
    args = parser.parse_args()
    if args.report:
        path = os.path.join(PROJECT_ROOT, "results", "preprocessing_summary.json")
        print(open(path, encoding="utf-8").read() if os.path.exists(path) else "no summary yet")
        return 0
    summary = run_preprocessing()
    print(json.dumps({k: v for k, v in summary.items() if k != "million"}, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
