"""Evaluation, plots and artifact export for the student-performance models.

    python src/evaluate.py                 # re-evaluate the saved final models on the test split
    python src/evaluate.py --dataset exams # only the secondary exams model

Everything here is computed from saved artifacts + the untouched test split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from preprocess import (MODEL_FEATURES, RAW_FEATURES, TARGET_COLUMN,  # noqa: E402
                        load_processed)
from utils import classification_metrics, regression_metrics, save_json  # noqa: E402

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "student_performance")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
VIS_DIR = os.path.join(RESULTS_DIR, "visualizations")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model", "model.joblib")
GRADE_MODEL_PATH = os.path.join(MODELS_DIR, "best_model", "grade_model.joblib")
EXAMS_MODEL_PATH = os.path.join(MODELS_DIR, "best_model", "exams_model.joblib")


def ensure_dirs() -> None:
    for path in (RESULTS_DIR, VIS_DIR, MODELS_DIR):
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def features_for(pre: Dict[str, Any], df: pd.DataFrame, scaled: bool) -> np.ndarray:
    X = df[MODEL_FEATURES].to_numpy(dtype=np.float64)
    return pre["scaler"].transform(X) if scaled else X


def evaluate_regression(artifact: Dict[str, Any], pre: Dict[str, Any],
                        df: pd.DataFrame) -> Dict[str, Any]:
    X = features_for(pre, df, artifact["scaled"])
    y_pred = artifact["model"].predict(X)
    y_true = df[TARGET_COLUMN].to_numpy(dtype=float)
    metrics = regression_metrics(y_true, y_pred)
    metrics["predictions_outside_0_100"] = int(((y_pred < 0) | (y_pred > 100)).sum())
    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred}


def evaluate_classifier(artifact: Dict[str, Any], pre: Dict[str, Any],
                        df: pd.DataFrame) -> Dict[str, Any]:
    model = artifact["model"]
    X = df[artifact["features"]].to_numpy(dtype=np.float64)
    y_idx = df["grade_index"].to_numpy()
    proba = model.predict_proba(X)
    y_pred = model.predict(X)
    labels = list(range(len(artifact["labels"])))
    metrics = classification_metrics(y_idx, y_pred, y_proba=proba, labels=labels)
    metrics["label_names"] = artifact["labels"]
    return {"metrics": metrics, "y_true": y_idx, "y_pred": y_pred, "proba": proba}


def evaluate_exams(artifact: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "processed", "tabular",
                                      "exams_test.parquet"))
    X = artifact["encoder"].transform(df[artifact["features"]])
    y_pred = np.clip(artifact["model"].predict(X), 0, 100)
    y_true = df["math score"].to_numpy(dtype=float)
    metrics = regression_metrics(y_true, y_pred)
    # Optional coarse categories with the documented thresholds.
    cut = artifact["category_cutoffs"]

    def cat(v):
        if v < cut["needs improvement"]:
            return 0
        if v < cut["satisfactory"]:
            return 1
        return 2

    y_true_c = np.array([cat(v) for v in y_true])
    y_pred_c = np.array([cat(v) for v in y_pred])
    metrics["category_accuracy"] = float((y_true_c == y_pred_c).mean())
    metrics["category_labels"] = ["needs improvement (<60)", "satisfactory (60-79)", "good (>=80)"]
    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred,
            "categories": (y_true_c, y_pred_c)}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _save(fig, name: str) -> str:
    ensure_dirs()
    path = os.path.join(VIS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_actual_vs_predicted(y_true, y_pred, title: str, name: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.15, color="#2b6cb0", edgecolors="none")
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "--", color="#c53030", linewidth=1.2, label="perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, name)


def plot_residuals(y_true, y_pred, name: str = "residuals.png") -> str:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred, residuals, s=4, alpha=0.15, color="#2f855a", edgecolors="none")
    axes[0].axhline(0, color="#c53030", linewidth=1.2)
    axes[0].set_xlabel("Predicted score")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs predictions")
    axes[0].grid(alpha=0.25)
    axes[1].hist(residuals, bins=80, color="#4a5568")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual distribution")
    axes[1].grid(alpha=0.25)
    return _save(fig, name)


def plot_feature_importance(importance: Dict[str, float], name: str = "feature_importance.png") -> str:
    items = sorted(importance.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#2b6cb0")
    ax.set_xlabel("Permutation importance (increase in RMSE)")
    ax.set_title("Feature importance on the test split")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, name)


def plot_model_comparison(rows: List[Dict[str, Any]], name: str = "model_comparison.png") -> str:
    names = [r["name"] for r in rows]
    mae = [r["val_metrics"]["mae"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, mae, color="#4299e1")
    ax.set_ylabel("Validation MAE (lower is better)")
    ax.set_title("Candidate model comparison")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(mae):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    return _save(fig, name)


def plot_confusion_matrix(cm: List[List[int]], labels: List[str],
                          name: str = "confusion_matrix.png") -> str:
    cm_arr = np.asarray(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    normalized = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("Actual grade")
    ax.set_title("Grade classifier — row-normalised confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{normalized[i, j]:.2f}\n({int(cm_arr[i, j]):,})",
                    ha="center", va="center", fontsize=7,
                    color="white" if normalized[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, name)


def plot_training_curves(curves: Dict[str, Any], name: str = "training_curves.png") -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plotted = False
    for label, values in curves.items():
        if values:
            axes[0].plot(range(1, len(values) + 1), values, label=label)
            plotted = True
    axes[0].set_xlabel("Epoch / iteration")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("Neural network loss curve")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    hist = curves.get("hist_gb_validation")
    if hist:
        iterations, scores = zip(*hist)
        axes[1].plot(iterations, scores, color="#c05621", label="validation loss")
        axes[1].set_xlabel("Boosting iteration")
        axes[1].set_ylabel("Validation loss (squared error)")
        axes[1].set_title("Gradient boosting validation curve")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.25)
    else:
        axes[1].axis("off")
    if not plotted:
        axes[0].axis("off")
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Model loading / main
# ---------------------------------------------------------------------------
def load_artifact(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing artifact {path}; run `python src/train_model.py` first")
    return joblib.load(path)


def load_preprocessor_artifact() -> Dict[str, Any]:
    return joblib.load(os.path.join(MODELS_DIR, "preprocessor", "million_preprocessor.joblib"))


def recompute_test_metrics(write_plots: bool = True) -> Dict[str, Any]:
    ensure_dirs()
    pre = load_preprocessor_artifact()
    artifact = load_artifact(BEST_MODEL_PATH)
    test = load_processed("million")["test"]
    reg = evaluate_regression(artifact, pre, test)

    out: Dict[str, Any] = {
        "model_name": artifact["name"],
        "model_family": artifact.get("family"),
        "test_metrics": reg["metrics"],
        "n_features": len(MODEL_FEATURES),
        "features": MODEL_FEATURES,
        "target": TARGET_COLUMN,
        "evaluated_samples": int(len(test)),
    }
    if write_plots:
        out["plots"] = [
            plot_actual_vs_predicted(reg["y_true"], reg["y_pred"],
                                     f"{artifact['name']} — test split", "actual_vs_predicted.png"),
            plot_residuals(reg["y_true"], reg["y_pred"]),
        ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the trained student-performance models.")
    parser.add_argument("--dataset", choices=["million", "exams", "all"], default="all")
    args = parser.parse_args()
    results: Dict[str, Any] = {}
    if args.dataset in ("million", "all"):
        results["million_test"] = recompute_test_metrics()
        print(json.dumps(results["million_test"]["test_metrics"], indent=2))
    if args.dataset in ("exams", "all") and os.path.exists(EXAMS_MODEL_PATH):
        artifact = load_artifact(EXAMS_MODEL_PATH)
        exam = evaluate_exams(artifact)
        results["exams_test"] = {"metrics": exam["metrics"], "model_name": artifact["name"]}
        print(json.dumps(exam["metrics"], indent=2))
    if results:
        path = os.path.join(RESULTS_DIR, "evaluation_rerun.json")
        save_json(results, path)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
