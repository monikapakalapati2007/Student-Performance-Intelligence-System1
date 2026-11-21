"""End-to-end training pipeline for the Student Performance models.

    python src/train_model.py                 # full run (sanity check -> search -> train -> test)
    python src/train_model.py --quick         # small subsamples, for a fast smoke run
    python src/train_model.py --skip-sanity   # only if the sanity check already passed
    python src/train_model.py --reports-only  # rebuild training_summary.md from saved results

Pipeline: load config -> locate/validate datasets -> preprocess -> sanity check ->
candidate training + validation -> hyper-parameter search -> best-model selection ->
checkpointing -> refit on train+validation -> untouched-test evaluation -> grade
classifier -> secondary exams model -> sample inference -> summary.

The models are gradient-boosted trees / random forest / ridge / MLP on tabular data.
They are NOT language models; the natural-language demo in src/nl_demo.py is a
retrieval/template layer over these trained models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import evaluate as ev  # noqa: E402
from preprocess import (GRADE_COLUMN, MODEL_FEATURES, RAW_FEATURES, TARGET_COLUMN,  # noqa: E402
                        build_exam_preprocessor, load_exam_splits, load_million_splits,
                        save_single_preprocessor, transform_exams, transform_million,
                        write_artifacts)
from utils import (classification_metrics, format_seconds, get_hardware_info,  # noqa: E402
                   regression_metrics, save_json, set_seed)

MODELS_DIR = ev.MODELS_DIR
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
BEST_DIR = os.path.join(MODELS_DIR, "best_model")
FINAL_DIR = os.path.join(MODELS_DIR, "final_model")
RESULTS_DIR = ev.RESULTS_DIR
CONFIG_PATH = os.path.join(PROJECT_ROOT, "training_config.json")
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "training_summary.md")

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 1337,
    "dataset_selection": {
        "primary": "million",
        "secondary": "exams",
        "reason": ("the 1M-record table has 3 behavioural features plus a continuous score and a "
                   "grade label; the 1k exams table has 5 categorical features and three exam "
                   "scores. They share no columns and no key, so they are trained separately."),
    },
    "tasks": {
        "regression": {"target": TARGET_COLUMN, "features": MODEL_FEATURES},
        "classification": {"target": GRADE_COLUMN, "features": MODEL_FEATURES,
                           "note": "total_score is never used as an input feature"},
    },
    "sanity_check": {"n_train": 5000, "n_eval": 2000, "max_iter": 25,
                     "mlp_max_iter": 2, "mlp_rows": 8000},
    "candidates": {
        "ridge": {"enabled": True, "alpha": 1.0},
        "random_forest": {"enabled": True, "n_estimators": 120, "max_depth": 18,
                          "min_samples_leaf": 20, "max_samples": 300000},
        "hist_gradient_boosting": {"enabled": True, "max_iter": 400, "learning_rate": 0.06,
                                   "max_leaf_nodes": 31, "min_samples_leaf": 40,
                                   "l2_regularization": 1.0, "early_stopping": True,
                                   "n_iter_no_change": 20, "validation_fraction": 0.1},
        "mlp": {"enabled": True, "hidden_layer_sizes": [64, 32], "alpha": 0.001,
                "batch_size": 512, "learning_rate_init": 0.001, "max_iter": 40,
                "early_stopping": True, "n_iter_no_change": 6, "max_samples": 200000},
    },
    "hyperparameter_search": {
        "enabled": True,
        "max_trials": 3,
        "hist_gradient_boosting": [
            {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31,
             "min_samples_leaf": 40, "l2_regularization": 1.0},
            {"max_iter": 300, "learning_rate": 0.08, "max_leaf_nodes": 63,
             "min_samples_leaf": 20, "l2_regularization": 0.5},
            {"max_iter": 500, "learning_rate": 0.04, "max_leaf_nodes": 15,
             "min_samples_leaf": 80, "l2_regularization": 2.0},
        ],
    },
    "grade_classifier": {
        "candidates": ["hist_gradient_boosting", "random_forest"],
        "hist_gradient_boosting": {"max_iter": 200, "learning_rate": 0.08,
                                   "max_leaf_nodes": 31, "min_samples_leaf": 40,
                                   "early_stopping": True, "n_iter_no_change": 15},
        "random_forest": {"n_estimators": 80, "max_depth": 16, "min_samples_leaf": 20,
                          "class_weight": "balanced_subsample", "max_samples": 300000},
    },
    "exams_model": {
        "enabled": True,
        "target": "math score",
        "max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15,
        "l2_regularization": 1.0, "early_stopping": True, "n_iter_no_change": 20,
    },
    "inference": {"sample_rows": 5, "importance_rows": 15000, "importance_repeats": 3},
}


# ---------------------------------------------------------------------------
# Config / setup
# ---------------------------------------------------------------------------
def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        save_json(DEFAULT_CONFIG, path)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    for key, value in cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def ensure_dirs() -> None:
    for path in (MODELS_DIR, CHECKPOINT_DIR, BEST_DIR, FINAL_DIR, RESULTS_DIR, ev.VIS_DIR,
                 os.path.join(MODELS_DIR, "preprocessor")):
        os.makedirs(path, exist_ok=True)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def checkpoint(model: Any, name: str, meta: Dict[str, Any]) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"{name}.joblib")
    joblib.dump({"model": model, "meta": meta}, path)
    return path


def load_checkpoint(path: str) -> Dict[str, Any]:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def build_regressor(name: str, params: Dict[str, Any], seed: int):
    if name == "ridge":
        return Ridge(alpha=params.get("alpha", 1.0))
    if name == "linear_baseline":
        return LinearRegression()
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 120),
            max_depth=params.get("max_depth", 18),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            max_features=params.get("max_features", 1.0),
            n_jobs=params.get("n_jobs", -1),
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=params.get("max_iter", 400),
            learning_rate=params.get("learning_rate", 0.06),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            min_samples_leaf=params.get("min_samples_leaf", 40),
            l2_regularization=params.get("l2_regularization", 1.0),
            early_stopping=params.get("early_stopping", True),
            n_iter_no_change=params.get("n_iter_no_change", 20),
            validation_fraction=params.get("validation_fraction", 0.1),
            random_state=seed,
        )
    if name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (64, 32))),
            activation="relu",
            solver="adam",
            alpha=params.get("alpha", 0.001),
            batch_size=params.get("batch_size", 512),
            learning_rate_init=params.get("learning_rate_init", 0.001),
            max_iter=params.get("max_iter", 40),
            early_stopping=params.get("early_stopping", True),
            n_iter_no_change=params.get("n_iter_no_change", 6),
            validation_fraction=0.1,
            random_state=seed,
        )
    raise ValueError(f"unknown regressor {name}")


def needs_scaling(name: str) -> bool:
    return name in {"ridge", "linear_baseline", "mlp"}


def subsample(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if not cap or len(df) <= cap:
        return df
    return df.sample(n=cap, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
def sanity_check(cfg: Dict[str, Any], pre: Dict[str, Any],
                 splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Fast end-to-end rehearsal on small samples. Raises on any failure."""
    log("\n[2/8] Sanity check (small samples, no full training yet)")
    sc = cfg["sanity_check"]
    seed = cfg["seed"]
    checks: List[Dict[str, Any]] = []

    def record(name: str, ok: bool, detail: Any = "") -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        log(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
        if not ok:
            raise RuntimeError(f"sanity check failed: {name} ({detail})")

    train = splits["train"]
    val = splits["validation"]
    train_small = subsample(train, sc["n_train"], seed)
    val_small = subsample(val, sc["n_eval"], seed)

    record("datasets loaded", len(train) > 0 and len(val) > 0,
           f"train={len(train):,} validation={len(val):,} test={len(splits['test']):,}")
    record("features exclude target",
           TARGET_COLUMN not in MODEL_FEATURES and GRADE_COLUMN not in MODEL_FEATURES,
           f"{len(MODEL_FEATURES)} features")
    record("no missing values in feature/target columns",
           bool(train_small[MODEL_FEATURES + [TARGET_COLUMN]].notna().all().all()))
    X_tr = transform_million(pre, train_small, scaled=True)
    X_va = transform_million(pre, val_small, scaled=True)
    record("preprocessor shapes", X_tr.shape[0] == len(train_small) and
           X_tr.shape[1] == len(MODEL_FEATURES), f"{X_tr.shape} / {X_va.shape}")
    record("preprocessed values finite", bool(np.isfinite(X_tr).all() and np.isfinite(X_va).all()))

    model = build_regressor("hist_gradient_boosting",
                            {"max_iter": sc["max_iter"], "early_stopping": False}, seed)
    model.fit(X_tr, train_small[TARGET_COLUMN].to_numpy())
    pred = model.predict(X_va)
    record("predictions shape", pred.shape == (len(val_small),), f"{pred.shape}")
    metrics = regression_metrics(val_small[TARGET_COLUMN].to_numpy(), pred)
    record("validation metrics finite", all(np.isfinite(v) for v in metrics.values()),
           f"MAE={metrics['mae']:.3f} R2={metrics['r2']:.4f}")

    # Checkpoint round-trip.
    tmp_path = checkpoint(model, "_sanity_tmp", {"name": "sanity"})
    reloaded = load_checkpoint(tmp_path)
    replay = reloaded["model"].predict(X_va)
    record("checkpoint save/load reproduces predictions", bool(np.allclose(pred, replay)),
           f"{os.path.basename(tmp_path)}")
    os.remove(tmp_path)

    # Neural-network path: one short fit must reduce its own loss and be resumable.
    mlp = build_regressor("mlp", {"hidden_layer_sizes": [16], "max_iter": sc["mlp_max_iter"],
                                  "batch_size": 256, "early_stopping": False}, seed)
    mlp.fit(X_tr, train_small[TARGET_COLUMN].to_numpy())
    curve = [float(v) for v in mlp.loss_curve_]
    record("MLP training executed, finite loss", bool(curve) and np.isfinite(curve).all(),
           f"{len(curve)} iterations, first={curve[0]:.2f} last={curve[-1]:.2f}")
    record("MLP loss decreases", curve[-1] < curve[0], f"{curve[0]:.2f} -> {curve[-1]:.2f}")

    # Grade classifier can be built and scored.
    g = HistGradientBoostingClassifier(max_iter=20, early_stopping=False, random_state=seed)
    g.fit(train_small[MODEL_FEATURES].to_numpy(), train_small["grade_index"].to_numpy())
    g_pred = g.predict(val_small[MODEL_FEATURES].to_numpy())
    record("grade classifier predicts known labels",
           set(np.unique(g_pred)) <= set(train_small["grade_index"].unique()),
           f"accuracy={(g_pred == val_small['grade_index'].to_numpy()).mean():.3f}")

    log(f"  sanity check passed: {len(checks)}/{len(checks)} checks")
    return {"checks": checks, "passed": True,
            "sample_mae": metrics["mae"], "sample_r2": metrics["r2"],
            "rows_used": {"train": int(len(train_small)), "validation": int(len(val_small))}}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_candidate(name: str, params: Dict[str, Any], pre: Dict[str, Any],
                    train: pd.DataFrame, val: pd.DataFrame, seed: int) -> Dict[str, Any]:
    scaled = needs_scaling(name)
    train_used = subsample(train, params.get("max_samples", 0), seed) if name in {
        "random_forest", "mlp"} else train
    X_tr = transform_million(pre, train_used, scaled=scaled)
    X_va = transform_million(pre, val, scaled=scaled)
    y_tr = train_used[TARGET_COLUMN].to_numpy(dtype=float)
    y_va = val[TARGET_COLUMN].to_numpy(dtype=float)

    model = build_regressor(name, params, seed)
    started = time.time()
    model.fit(X_tr, y_tr)
    fit_seconds = time.time() - started
    pred = model.predict(X_va)
    metrics = regression_metrics(y_va, pred)
    return {"name": name, "family": name, "params": params, "model": model,
            "scaled": scaled, "n_train_used": int(len(train_used)),
            "fit_seconds": round(fit_seconds, 2), "val_metrics": metrics}


def evaluate_on_test(best: Dict[str, Any], pre: Dict[str, Any],
                     test: pd.DataFrame) -> Dict[str, Any]:
    res = ev.evaluate_regression(best, pre, test)
    return {"metrics": res["metrics"], "y_true": res["y_true"], "y_pred": res["y_pred"]}


# ---------------------------------------------------------------------------
# Grade classifier
# ---------------------------------------------------------------------------
def build_grade_classifier(name: str, params: Dict[str, Any], seed: int):
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=params.get("max_iter", 200),
            learning_rate=params.get("learning_rate", 0.08),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            min_samples_leaf=params.get("min_samples_leaf", 40),
            early_stopping=params.get("early_stopping", True),
            n_iter_no_change=params.get("n_iter_no_change", 15),
            validation_fraction=0.1,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 80),
            max_depth=params.get("max_depth", 16),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            class_weight=params.get("class_weight", "balanced_subsample"),
            n_jobs=params.get("n_jobs", -1),
            random_state=seed,
        )
    raise ValueError(name)


def train_grade_models(cfg: Dict[str, Any], pre: Dict[str, Any], train: pd.DataFrame,
                       val: pd.DataFrame, test: pd.DataFrame) -> Dict[str, Any]:
    seed = cfg["seed"]
    gcfg = cfg["grade_classifier"]
    labels = pre["grade_labels"]
    results = []
    for name in gcfg["candidates"]:
        params = dict(gcfg.get(name, {}))
        model = build_grade_classifier(name, params, seed)
        train_used = subsample(train, params.get("max_samples", 0), seed) if name == "random_forest" else train
        started = time.time()
        model.fit(train_used[MODEL_FEATURES].to_numpy(), train_used["grade_index"].to_numpy())
        fit_seconds = time.time() - started
        proba = model.predict_proba(val[MODEL_FEATURES].to_numpy())
        pred = model.predict(val[MODEL_FEATURES].to_numpy())
        metrics = classification_metrics(val["grade_index"].to_numpy(), pred, y_proba=proba,
                                         labels=list(range(len(labels))))
        results.append({"name": name, "params": params, "model": model, "scaled": False,
                        "features": MODEL_FEATURES, "labels": labels,
                        "fit_seconds": round(fit_seconds, 2), "val_metrics": metrics,
                        "n_train_used": int(len(train_used))})
        checkpoint(model, f"grade_{name}", {"name": name, "labels": labels,
                                            "features": MODEL_FEATURES,
                                            "val_metrics": metrics, "params": params})
        log(f"  {name:<26} val macro-F1={metrics['f1_macro']:.4f} "
            f"accuracy={metrics['accuracy']:.4f} acc_weighted={metrics['f1_weighted']:.4f} "
            f"({fit_seconds:.1f}s)")

    # Weighted F1 is the selection criterion: it reflects per-student accuracy under the
    # strong class imbalance (grade F is only 0.6 % of rows), while macro-F1 is still
    # reported. Accuracy breaks ties.
    best = max(results, key=lambda r: (r["val_metrics"]["f1_weighted"],
                                       r["val_metrics"]["accuracy"]))
    log(f"  selected grade classifier: {best['name']} "
        f"(val weighted-F1={best['val_metrics']['f1_weighted']:.4f}, "
        f"accuracy={best['val_metrics']['accuracy']:.4f})")

    # Refit on train + validation, then score the untouched test split.
    combined = pd.concat([train, val], ignore_index=True)
    combined_used = subsample(combined, best["params"].get("max_samples", 0), seed) \
        if best["name"] == "random_forest" else combined
    final_model = build_grade_classifier(best["name"], best["params"], seed)
    final_model.fit(combined_used[MODEL_FEATURES].to_numpy(),
                    combined_used["grade_index"].to_numpy())
    proba = final_model.predict_proba(test[MODEL_FEATURES].to_numpy())
    pred = final_model.predict(test[MODEL_FEATURES].to_numpy())
    test_metrics = classification_metrics(test["grade_index"].to_numpy(), pred, y_proba=proba,
                                          labels=list(range(len(labels))))
    test_metrics["label_names"] = labels

    artifact = {"model": final_model, "name": best["name"], "family": best["name"],
                "scaled": False, "features": MODEL_FEATURES, "labels": labels,
                "params": best["params"], "val_metrics": best["val_metrics"],
                "test_metrics": test_metrics, "n_train_used": int(len(combined_used))}
    joblib.dump(artifact, ev.GRADE_MODEL_PATH)
    return {"selected": best["name"], "candidates": [
        {k: v for k, v in r.items() if k != "model"} for r in results],
        "test_metrics": test_metrics,
        "predictions": {"y_true": test["grade_index"].to_numpy(), "y_pred": pred, "proba": proba},
        "labels": labels, "n_train_used": int(len(combined_used))}


# ---------------------------------------------------------------------------
# Secondary exams model
# ---------------------------------------------------------------------------
def train_exams_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    log("\n[7/8] Secondary model: Students Performance in Exams (math score)")
    ecfg = cfg["exams_model"]
    if not ecfg.get("enabled", True):
        return {"enabled": False}
    seed = cfg["seed"]
    splits, log_info = load_exam_splits()
    pre = build_exam_preprocessor(splits["train"])
    save_single_preprocessor(pre)
    write_artifacts(splits, "exams")
    X_tr = transform_exams(pre, splits["train"])
    X_va = transform_exams(pre, splits["validation"])
    X_te = transform_exams(pre, splits["test"])
    y_tr = splits["train"]["math score"].to_numpy(dtype=float)
    y_va = splits["validation"]["math score"].to_numpy(dtype=float)
    y_te = splits["test"]["math score"].to_numpy(dtype=float)

    # Two models on a 700-row training split: a linear baseline and a small booster.
    lin = LinearRegression().fit(X_tr, y_tr)
    lin_val = regression_metrics(y_va, np.clip(lin.predict(X_va), 0, 100))
    gb = HistGradientBoostingRegressor(
        max_iter=ecfg.get("max_iter", 300), learning_rate=ecfg.get("learning_rate", 0.05),
        max_leaf_nodes=ecfg.get("max_leaf_nodes", 15),
        l2_regularization=ecfg.get("l2_regularization", 1.0),
        early_stopping=ecfg.get("early_stopping", True),
        n_iter_no_change=ecfg.get("n_iter_no_change", 20),
        validation_fraction=0.15, random_state=seed,
    ).fit(X_tr, y_tr)
    gb_val = regression_metrics(y_va, np.clip(gb.predict(X_va), 0, 100))
    log(f"  linear baseline val R2={lin_val['r2']:.4f} MAE={lin_val['mae']:.2f}")
    log(f"  hist gradient boosting val R2={gb_val['r2']:.4f} MAE={gb_val['mae']:.2f}")
    chosen, chosen_name = (gb, "hist_gradient_boosting") if gb_val["rmse"] <= lin_val["rmse"] \
        else (lin, "linear_regression")

    combined = pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    X_all = transform_exams(pre, combined)
    y_all = combined["math score"].to_numpy(dtype=float)
    final = (HistGradientBoostingRegressor(
        max_iter=ecfg.get("max_iter", 300), learning_rate=ecfg.get("learning_rate", 0.05),
        max_leaf_nodes=ecfg.get("max_leaf_nodes", 15),
        l2_regularization=ecfg.get("l2_regularization", 1.0),
        early_stopping=ecfg.get("early_stopping", True),
        n_iter_no_change=ecfg.get("n_iter_no_change", 20),
        validation_fraction=0.1, random_state=seed).fit(X_all, y_all)
        if chosen_name == "hist_gradient_boosting" else LinearRegression().fit(X_all, y_all))

    artifact = {"model": final, "name": chosen_name, "family": chosen_name,
                "encoder": pre["encoder"], "features": pre["categorical_features"],
                "feature_names": pre["feature_names"], "target": "math score",
                "category_cutoffs": pre["category_cutoffs"],
                "val_metrics": {"linear": lin_val, chosen_name: gb_val if chosen_name ==
                                "hist_gradient_boosting" else gb_val}}
    ev.ensure_dirs()
    joblib.dump(artifact, ev.EXAMS_MODEL_PATH)
    exam_eval = ev.evaluate_exams(artifact)
    test_metrics = exam_eval["metrics"]
    log(f"  exams test: MAE={test_metrics['mae']:.2f} RMSE={test_metrics['rmse']:.2f} "
        f"R2={test_metrics['r2']:.4f} (n={test_metrics['n_samples']})")
    return {"enabled": True, "selected": chosen_name,
            "candidates": {"linear_regression": lin_val, "hist_gradient_boosting": gb_val},
            "test_metrics": test_metrics, "preprocessing": log_info,
            "predictions": {"y_true": exam_eval["y_true"].tolist(),
                            "y_pred": exam_eval["y_pred"].tolist()},
            "features": pre["categorical_features"], "n_train_used": int(len(combined))}


# ---------------------------------------------------------------------------
# Inference samples + summary
# ---------------------------------------------------------------------------
def sample_inference(cfg: Dict[str, Any], pre: Dict[str, Any], artifact: Dict[str, Any],
                     grade_artifact: Dict[str, Any], test: pd.DataFrame,
                     exams_artifact: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    log("\n[8/8] Sample inference")
    rows = test.sample(n=cfg["inference"]["sample_rows"], random_state=cfg["seed"])
    scaled = artifact["scaled"]
    preds = np.clip(artifact["model"].predict(ev.features_for(pre, rows, scaled)), 0, 100)
    grade_proba = grade_artifact["model"].predict_proba(
        rows[grade_artifact["features"]].to_numpy())
    grade_labels = grade_artifact["labels"]
    samples = []
    for (_, row), pred, proba in zip(rows.iterrows(), preds, grade_proba):
        idx = int(np.argmax(proba))
        samples.append({
            "student_id": int(row["student_id"]) if "student_id" in row else None,
            "input": {c: float(row[c]) for c in RAW_FEATURES},
            "actual_total_score": float(row[TARGET_COLUMN]),
            "predicted_total_score": round(float(pred), 2),
            "actual_grade": str(row[GRADE_COLUMN]),
            "predicted_grade": grade_labels[idx],
            "grade_confidence": round(float(proba[idx]), 3),
        })
    for s in samples:
        log(f"  id={s['student_id']} inputs={ {k: round(v, 2) for k, v in s['input'].items()} } "
            f"-> predicted {s['predicted_total_score']} (actual {s['actual_total_score']}), "
            f"grade {s['predicted_grade']} (actual {s['actual_grade']})")

    # Illustrative, clearly synthetic inputs (not dataset records) to show the API.
    hypothetical = [
        {"weekly_self_study_hours": 8.0, "attendance_percentage": 80.0,
         "class_participation": 5.0},
        {"weekly_self_study_hours": 18.0, "attendance_percentage": 92.0,
         "class_participation": 8.0},
        {"weekly_self_study_hours": 30.0, "attendance_percentage": 70.0,
         "class_participation": 3.0},
    ]
    hypo_df = pd.DataFrame(hypothetical)
    from preprocess import engineer_features

    hypo_df = engineer_features(hypo_df)
    hypo_pred = np.clip(artifact["model"].predict(ev.features_for(pre, hypo_df, scaled)), 0, 100)
    hypo_grade = grade_artifact["model"].predict_proba(hypo_df[grade_artifact["features"]].to_numpy())
    hypothetical_out = []
    for spec, pred, proba in zip(hypothetical, hypo_pred, hypo_grade):
        idx = int(np.argmax(proba))
        hypothetical_out.append({**spec, "predicted_total_score": round(float(pred), 2),
                                 "predicted_grade": grade_labels[idx],
                                 "grade_confidence": round(float(proba[idx]), 3)})
        log(f"  [illustrative input] {spec} -> {float(pred):.1f} ({grade_labels[idx]})")

    exams_sample = None
    if exams_artifact is not None:
        exams_test = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "processed", "tabular",
                                                  "exams_test.parquet"))
        row = exams_test.iloc[0]
        X = exams_artifact["encoder"].transform(pd.DataFrame([row[exams_artifact["features"]]]))
        exams_sample = {
            "input": {c: str(row[c]) for c in exams_artifact["features"]},
            "predicted_math_score": round(float(np.clip(exams_artifact["model"].predict(X)[0], 0, 100)), 2),
            "actual_math_score": float(row["math score"]),
        }
        log(f"  [exams] {exams_sample['input']} -> predicted math "
            f"{exams_sample['predicted_math_score']} (actual {exams_sample['actual_math_score']})")

    return {"test_samples": samples, "illustrative_inputs": hypothetical_out,
            "exams_sample": exams_sample}


def write_summary(results: Dict[str, Any]) -> None:
    hw = results["hardware"]
    sel = results["selected_model"]
    test = results["test_metrics"]
    grade = results["grade_classifier"]
    lines = [
        "# Training summary — Student Performance Prediction", "",
        f"Generated by `python src/train_model.py` on {results['finished_at']} "
        f"(duration {results['duration_human']}).", "",
        "## Environment", "",
        f"- CPU: {hw['cpu_name']} ({hw['cpu_count']} logical cores), RAM {hw['total_ram_gb']} GB",
        f"- GPU: {hw['gpu_names'] or 'none'} — CUDA available to PyTorch: {hw['cuda_available']}"
        + (" (GPU present but the installed PyTorch build is CPU-only)"
           if hw.get("gpu_detected_but_unusable") else ""),
        f"- Python {hw['python_version']}, PyTorch {hw['torch_version']}, "
        f"scikit-learn {hw.get('package_versions', {}).get('sklearn')}",
        f"- Compute device for training: {hw['selected_device']}", "",
        "## Data", "",
        f"- Primary: 1M-record student-performance table; train {results['splits']['train']:,}, "
        f"validation {results['splits']['validation']:,}, test {results['splits']['test']:,}",
        f"- Features ({len(MODEL_FEATURES)}): {', '.join(MODEL_FEATURES)}",
        "- Target: `total_score` (regression); `grade` (classification, behavioural features only)",
        "- The exams table is incompatible (different schema/targets) and is trained separately.", "",
        "## Sanity check", "",
    ]
    for c in results["sanity_checks"]["checks"]:
        lines.append(f"- {'PASS' if c['passed'] else 'FAIL'} — {c['check']}"
                     + (f" ({c['detail']})" if c["detail"] else ""))
    lines += ["", "## Candidate comparison (validation split)", "",
              "| model | train rows | val MAE | val RMSE | val R² | fit time |",
              "|---|---|---|---|---|---|"]
    for r in results["candidates"]:
        vm = r["val_metrics"]
        lines.append(f"| {r['name']} | {r['n_train_used']:,} | {vm['mae']:.3f} | {vm['rmse']:.3f} | "
                     f"{vm['r2']:.4f} | {r['fit_seconds']:.1f}s |")
    lines += ["", f"Selected on validation RMSE: **{sel['name']}** "
                  f"({sel['params']}).", "",
              "## Held-out test metrics (final model, train+validation refit)", "",
              f"- MAE: **{test['mae']:.3f}** points", f"- MSE: {test['mse']:.3f}",
              f"- RMSE: **{test['rmse']:.3f}** points", f"- R²: **{test['r2']:.4f}**",
              f"- Test rows: {test['n_samples']:,}", "",
              "## Grade classifier (behavioural features only)", "",
              f"- Selected: {grade['selected']}",
              f"- Test accuracy: **{grade['test_metrics']['accuracy']:.4f}**",
              f"- Test macro-F1: **{grade['test_metrics']['f1_macro']:.4f}**",
              f"- Test weighted-F1: {grade['test_metrics']['f1_weighted']:.4f}",
              f"- ROC-AUC (OVR macro): {grade['test_metrics'].get('roc_auc_ovr_macro', 'n/a')}", "",
              "### Confusion matrix (rows = actual, columns = predicted)", "",
              "| actual \\ predicted | " + " | ".join(grade["labels"]) + " |",
              "|---|" + "---|" * len(grade["labels"])]
    for label, row in zip(grade["labels"], grade["test_metrics"]["confusion_matrix"]):
        lines.append(f"| {label} | " + " | ".join(f"{v:,}" for v in row) + " |")
    if results.get("exams_model", {}).get("enabled"):
        ex = results["exams_model"]["test_metrics"]
        lines += ["", "## Secondary model — exams math score", "",
                  f"- Selected: {results['exams_model']['selected']}",
                  f"- Test MAE: {ex['mae']:.3f}, RMSE: {ex['rmse']:.3f}, R²: {ex['r2']:.4f} "
                  f"(n={ex['n_samples']})",
                  f"- Coarse category accuracy (<60 / 60-79 / 80+): "
                  f"{ex.get('category_accuracy', 0):.4f}",
                  "- Only 700 rows of training data; this model is a secondary reference, "
                  "not the primary predictor."]
    lines += ["", "## Sample predictions", ""]
    for s in results["samples"]["test_samples"]:
        lines.append(f"- id {s['student_id']}: predicted {s['predicted_total_score']} "
                     f"(actual {s['actual_total_score']}), grade {s['predicted_grade']} "
                     f"(actual {s['actual_grade']})")
    lines += ["", "## Feature importance (permutation, test split)", ""]
    for k, v in sorted(results["feature_importance"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v:.4f}")
    lines += ["", "## Artifacts", ""]
    for k, v in results["artifacts"].items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Honest limitations", "",
              "- The target is right-censored at 100 points (26.8 % of rows), and the only "
              "strong predictor is weekly study hours (r = 0.81). Attendance and participation "
              "carry almost no signal in this synthetic table, so R² is bounded near 0.66.",
              "- Grade is a near-deterministic binning of the score (99.79 % agreement at the "
              "documented cut-offs); the grade model therefore learns the noisy boundary "
              "behaviour from behaviour features and is useful only as a coarse indicator.",
              "- These are tabular tree/MLP models, not language models. The natural-language "
              "demo is a template/retrieval layer over them, and it labels observations vs "
              "predictions explicitly.", ""]
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def reports_only() -> int:
    path = os.path.join(RESULTS_DIR, "run_results.json")
    if not os.path.exists(path):
        print("no saved run results; run the full pipeline first")
        return 1
    with open(path, encoding="utf-8") as fh:
        results = json.load(fh)
    write_summary(results)
    print(f"Rewrote {SUMMARY_PATH}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Train the student-performance models end to end.")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--quick", action="store_true", help="tiny subsamples for a smoke test")
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--reports-only", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=0,
                        help="cap training rows (0 = use the full provided split)")
    args = parser.parse_args()

    if args.reports_only:
        return reports_only()

    cfg = load_config(args.config)
    if args.quick:
        cfg["candidates"]["random_forest"]["max_samples"] = 20000
        cfg["candidates"]["mlp"]["max_samples"] = 20000
        cfg["candidates"]["hist_gradient_boosting"]["max_iter"] = 60
        cfg["hyperparameter_search"]["enabled"] = False
        cfg["grade_classifier"]["random_forest"]["max_samples"] = 20000
        cfg["grade_classifier"]["random_forest"]["n_estimators"] = 20
        cfg["grade_classifier"]["hist_gradient_boosting"]["max_iter"] = 40
        cfg["inference"]["importance_rows"] = 2000
    seed = cfg["seed"]
    set_seed(seed)
    ensure_dirs()
    started = time.time()

    log("=" * 78)
    log("Student Performance Prediction — automated training pipeline")
    log("=" * 78)

    # 1. Environment
    hw = get_hardware_info()
    hw["package_versions"] = {}
    for mod_name in ("numpy", "pandas", "sklearn", "joblib", "matplotlib", "pyarrow"):
        try:
            hw["package_versions"][mod_name] = getattr(__import__(mod_name), "__version__", "unknown")
        except Exception:
            hw["package_versions"][mod_name] = "missing"
    log("\n[1/8] Environment")
    log(f"  CPU: {hw['cpu_name']} ({hw['cpu_count']} cores), RAM {hw['total_ram_gb']} GB")
    log(f"  GPU: {hw['gpu_names'] or 'none'}, CUDA available: {hw['cuda_available']}")
    log(f"  Python {hw['python_version']}, PyTorch {hw['torch_version']}, device {hw['selected_device']}")
    log("  Note: the installed PyTorch wheel is CPU-only, and gradient-boosted trees used here")
    log("        train on CPU regardless, so the GPU provides no speed-up for this pipeline.")

    # 2. Data
    log("\n[1/8] Data loading and validation")
    splits, million_log = load_million_splits(verbose=True)
    required = {"train", "validation", "test"}
    if not required <= set(splits):
        raise RuntimeError(f"missing splits: {required - set(splits)}")
    if args.max_train_rows:
        splits["train"] = subsample(splits["train"], args.max_train_rows, seed)

    grade_map = {g: i for i, g in enumerate(["F", "D", "C", "B", "A"])}
    for split, df in splits.items():
        df["grade_index"] = df[GRADE_COLUMN].map(grade_map).astype(int)
        splits[split] = df

    # Always refit preprocessing/serialisation from the training split for reproducibility.
    from preprocess import build_million_preprocessor

    pre = build_million_preprocessor(splits["train"])
    save_single_preprocessor(pre)
    write_artifacts(splits, "million")
    log(f"  train {len(splits['train']):,} | validation {len(splits['validation']):,} | "
        f"test {len(splits['test']):,}")
    log(f"  features: {MODEL_FEATURES}")
    log(f"  preprocessing: {million_log['splits']}")

    # 3. Sanity
    sanity = {"skipped": True} if args.skip_sanity else sanity_check(cfg, pre, splits)
    if args.skip_sanity:
        log("\n[2/8] Sanity check skipped by request")

    # 4. Candidates
    log("\n[3/8] Candidate models (validation split)")
    train, val, test = splits["train"], splits["validation"], splits["test"]
    candidates: List[Dict[str, Any]] = []
    for name, params in cfg["candidates"].items():
        if not params.get("enabled", True):
            continue
        log(f"  training {name} ...")
        try:
            res = train_candidate(name, params, pre, train, val, seed)
        except Exception as exc:  # never skip silently
            log(f"  FAILED {name}: {exc}")
            traceback.print_exc()
            raise
        candidates.append(res)
        vm = res["val_metrics"]
        log(f"  {name:<26} MAE={vm['mae']:.3f} RMSE={vm['rmse']:.3f} R2={vm['r2']:.4f} "
            f"({res['fit_seconds']:.1f}s, {res['n_train_used']:,} rows)")
        checkpoint(res["model"], name, {"name": name, "params": params, "scaled": res["scaled"],
                                        "val_metrics": vm, "n_train_used": res["n_train_used"]})

    # 5. Hyper-parameter search for the strongest family
    search_log: List[Dict[str, Any]] = []
    if cfg["hyperparameter_search"].get("enabled", True):
        log("\n[4/8] Hyper-parameter search (validation MAE)")
        trials = cfg["hyperparameter_search"]["hist_gradient_boosting"][
            : cfg["hyperparameter_search"].get("max_trials", 3)]
        base = dict(cfg["candidates"]["hist_gradient_boosting"])
        for i, trial in enumerate(trials, 1):
            params = {**base, **trial}
            t0 = time.time()
            res = train_candidate("hist_gradient_boosting", params, pre, train, val, seed)
            entry = {"trial": i, "params": params, "val_metrics": res["val_metrics"],
                     "fit_seconds": round(time.time() - t0, 2)}
            search_log.append(entry)
            log(f"  trial {i}: MAE={res['val_metrics']['mae']:.3f} "
                f"R2={res['val_metrics']['r2']:.4f} {params}")
            candidates.append(res)
    else:
        log("\n[4/8] Hyper-parameter search disabled in config")

    # 6. Select best on validation RMSE
    log("\n[5/8] Best-model selection")
    best = min(candidates, key=lambda r: r["val_metrics"]["rmse"])
    log(f"  best: {best['name']} (validation RMSE {best['val_metrics']['rmse']:.3f})")

    # 7. Refit on train + validation, then test
    log("\n[6/8] Refit on train+validation and evaluate on the untouched test split")
    combined = pd.concat([train, val], ignore_index=True)
    refit_params = dict(best["params"])
    final = train_candidate(best["name"], refit_params, pre, combined, test, seed)
    test_eval = evaluate_on_test(final, pre, test)
    tm = test_eval["metrics"]
    log(f"  test: MAE={tm['mae']:.3f} RMSE={tm['rmse']:.3f} R2={tm['r2']:.4f} "
        f"(n={tm['n_samples']:,})")

    size_mb = None
    artifact = {"model": final["model"], "name": best["name"], "family": best["family"],
                "scaled": best["scaled"], "features": MODEL_FEATURES,
                "target": TARGET_COLUMN, "params": refit_params,
                "val_metrics": best["val_metrics"], "test_metrics": tm,
                "n_train_used": final["n_train_used"], "seed": seed}
    for directory in (BEST_DIR, FINAL_DIR):
        path = os.path.join(directory, "model.joblib")
        joblib.dump(artifact, path)
        size_mb = round(os.path.getsize(path) / (1024 ** 2), 3)
    checkpoint(final["model"], "final_refit", {"name": best["name"], "params": refit_params,
                                               "scaled": best["scaled"],
                                               "test_metrics": tm})
    save_json({k: v for k, v in artifact.items() if k != "model"},
              os.path.join(MODELS_DIR, "model_config.json"))

    # Grade classifier
    log("\n[6/8] Grade classifier (no score input)")
    grade_results = train_grade_models(cfg, pre, train, val, test)

    # Permutation importance on the final model
    log("\n  permutation importance (test split)")
    imp_rows = subsample(test, cfg["inference"]["importance_rows"], seed)
    perm = permutation_importance(final["model"], ev.features_for(pre, imp_rows, best["scaled"]),
                                  imp_rows[TARGET_COLUMN].to_numpy(), n_repeats=
                                  cfg["inference"]["importance_repeats"], random_state=seed,
                                  scoring="neg_root_mean_squared_error", n_jobs=1)
    importance = {name: float(v) for name, v in zip(MODEL_FEATURES, perm.importances_mean)}
    for k, v in sorted(importance.items(), key=lambda kv: -kv[1]):
        log(f"    {k:<32} {v:.4f}")

    # Exams model
    exams_results = train_exams_model(cfg)
    exams_artifact = None
    if exams_results.get("enabled") and os.path.exists(ev.EXAMS_MODEL_PATH):
        exams_artifact = joblib.load(ev.EXAMS_MODEL_PATH)

    samples = sample_inference(cfg, pre, artifact, joblib.load(ev.GRADE_MODEL_PATH), test,
                               exams_artifact)

    # Predictions CSV
    pred_df = pd.DataFrame({
        "student_id": test["student_id"].to_numpy(),
        **{c: test[c].to_numpy() for c in RAW_FEATURES},
        "actual_total_score": test_eval["y_true"],
        "predicted_total_score": np.round(test_eval["y_pred"], 3),
        "residual": np.round(test_eval["y_true"] - test_eval["y_pred"], 3),
        "actual_grade": test[GRADE_COLUMN].to_numpy(),
        "predicted_grade": [grade_results["labels"][i] for i in grade_results["predictions"]["y_pred"]],
        "predicted_grade_confidence": np.round(
            np.max(grade_results["predictions"]["proba"], axis=1), 4),
    })
    pred_path = os.path.join(RESULTS_DIR, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    # Visualisations
    ev.plot_actual_vs_predicted(test_eval["y_true"], test_eval["y_pred"],
                                f"{best['name']} — student total score (test)",
                                "actual_vs_predicted.png")
    ev.plot_residuals(test_eval["y_true"], test_eval["y_pred"])
    ev.plot_feature_importance(importance)
    ev.plot_model_comparison([{"name": r["name"], "val_metrics": r["val_metrics"]}
                              for r in candidates])
    ev.plot_confusion_matrix(grade_results["test_metrics"]["confusion_matrix"],
                             grade_results["labels"])
    curves: Dict[str, Any] = {}
    for res in candidates:
        if res["name"] == "mlp" and hasattr(res["model"], "loss_curve_"):
            curves["mlp_training_loss"] = [float(v) for v in res["model"].loss_curve_]
        if res["name"] == "hist_gradient_boosting" and hasattr(res["model"], "validation_score_"):
            hist = np.atleast_1d(res["model"].validation_score_)
            curves["hist_gb_validation"] = [(i + 1, float(v)) for i, v in enumerate(hist)]
    ev.plot_training_curves(curves)
    if exams_results.get("enabled"):
        ev.plot_actual_vs_predicted(np.asarray(exams_results["predictions"]["y_true"]),
                                    np.asarray(exams_results["predictions"]["y_pred"]),
                                    "Exams model — math score (test)", "exams_actual_vs_predicted.png")

    # Metrics file
    metrics_payload: Dict[str, Any] = {
        "primary_dataset": "Student Performance — 1M records",
        "task": "regression: total_score; classification: grade",
        "model": {"name": best["name"], "family": best["family"],
                  "params": refit_params, "scaled": best["scaled"],
                  "n_features": len(MODEL_FEATURES), "features": MODEL_FEATURES,
                  "n_train_used": final["n_train_used"],
                  "size_mb": size_mb, "seed": seed},
        "validation_metrics_selected": best["val_metrics"],
        "test_metrics": tm,
        "test_confusion_free_checks": {
            "predictions_equal_actual_rate": float(
                np.isclose(test_eval["y_true"], test_eval["y_pred"], atol=0.5).mean()),
        },
        "candidates": [{k: v for k, v in r.items() if k != "model"} for r in candidates],
        "hyperparameter_search": search_log,
        "feature_importance_permutation_rmse": importance,
        "grade_classifier": {
            "selected": grade_results["selected"],
            "candidates": grade_results["candidates"],
            "test_metrics": grade_results["test_metrics"],
            "labels": grade_results["labels"],
            "n_train_used": grade_results["n_train_used"],
        },
        "exams_model": {k: v for k, v in exams_results.items() if k != "predictions"},
        "samples": samples,
        "splits": {k: int(len(v)) for k, v in splits.items()},
        "hardware": hw,
    }
    save_json(metrics_payload, os.path.join(RESULTS_DIR, "metrics.json"))

    results = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(time.time() - started, 1),
        "duration_human": format_seconds(time.time() - started),
        "hardware": hw,
        "splits": metrics_payload["splits"],
        "sanity_checks": sanity,
        "candidates": metrics_payload["candidates"],
        "selected_model": {"name": best["name"], "params": refit_params,
                           "val_metrics": best["val_metrics"]},
        "test_metrics": tm,
        "grade_classifier": {"selected": grade_results["selected"],
                             "labels": grade_results["labels"],
                             "test_metrics": grade_results["test_metrics"]},
        "exams_model": {k: v for k, v in exams_results.items() if k != "predictions"},
        "samples": samples,
        "feature_importance": importance,
        "artifacts": {
            "best_model": os.path.relpath(os.path.join(BEST_DIR, "model.joblib"), PROJECT_ROOT),
            "final_model": os.path.relpath(os.path.join(FINAL_DIR, "model.joblib"), PROJECT_ROOT),
            "grade_model": os.path.relpath(ev.GRADE_MODEL_PATH, PROJECT_ROOT),
            "exams_model": os.path.relpath(ev.EXAMS_MODEL_PATH, PROJECT_ROOT),
            "checkpoints": os.path.relpath(CHECKPOINT_DIR, PROJECT_ROOT),
            "preprocessor": os.path.relpath(ev.MODELS_DIR + "/preprocessor", PROJECT_ROOT),
            "predictions": os.path.relpath(pred_path, PROJECT_ROOT),
            "metrics": "results/metrics.json",
            "visualizations": "results/visualizations",
        },
    }
    save_json(results, os.path.join(RESULTS_DIR, "run_results.json"))
    write_summary(results)

    log("\n" + "=" * 78)
    log(f"DONE in {results['duration_human']}")
    log(f"  best model      : {best['name']} (test MAE {tm['mae']:.2f}, R2 {tm['r2']:.4f})")
    log(f"  grade classifier: {grade_results['selected']} "
        f"(test macro-F1 {grade_results['test_metrics']['f1_macro']:.4f})")
    log(f"  saved models    : {MODELS_DIR}")
    log(f"  metrics         : {os.path.join(RESULTS_DIR, 'metrics.json')}")
    log(f"  summary         : {SUMMARY_PATH}")
    log("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
