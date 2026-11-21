"""Inference for the trained student-performance models.

    python src/predict.py --study-hours 15 --attendance 85 --participation 6.5
    python src/predict.py --json '{"weekly_self_study_hours": 18, "attendance_percentage": 92, "class_participation": 8}'
    python src/predict.py --input students.csv --out results/my_predictions.csv
    python src/predict.py --sample 5
    python src/predict.py --exams-json '{"gender": "female", "race/ethnicity": "group C", "parental level of education": "some college", "lunch": "standard", "test preparation course": "completed"}'

Loads the preprocessor and models from models/student_performance/ — nothing is
retrained here, so predictions are exactly reproducible from the saved artifacts.
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

from preprocess import MODEL_FEATURES, RAW_FEATURES, engineer_features  # noqa: E402

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "student_performance")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def _path(*parts: str) -> str:
    return os.path.join(MODELS_DIR, *parts)


class Predictor:
    """Tiny facade over the saved artifacts (regression + grade + exams models)."""

    def __init__(self, models_dir: str = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.pre = joblib.load(_path("preprocessor", "million_preprocessor.joblib")
                               if models_dir == MODELS_DIR else
                               os.path.join(models_dir, "preprocessor",
                                            "million_preprocessor.joblib"))
        self.reg = self._load("best_model", "model.joblib")
        self.grade = self._load("best_model", "grade_model.joblib")
        self.exams = self._load("best_model", "exams_model.joblib", required=False)

    def _load(self, *parts: str, required: bool = True) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.models_dir, *parts)
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"missing model artifact {path}; run "
                                        f"`python src/train_model.py` first")
            return None
        return joblib.load(path)

    # -- primary model -----------------------------------------------------
    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        for col in RAW_FEATURES:
            if col not in work.columns:
                raise ValueError(f"missing required column '{col}'")
            work[col] = pd.to_numeric(work[col], errors="coerce")
        if work[RAW_FEATURES].isna().any().any():
            raise ValueError("input contains missing or non-numeric values")
        work = engineer_features(work)
        X = work[MODEL_FEATURES].to_numpy(dtype=np.float64)
        if self.reg.get("scaled"):
            X = self.pre["scaler"].transform(X)
        scores = np.clip(self.reg["model"].predict(X), 0.0, 100.0)
        labels = self.grade["labels"]
        proba = self.grade["model"].predict_proba(work[self.grade["features"]].to_numpy())
        pred_idx = np.argmax(proba, axis=1)
        out = work[RAW_FEATURES].copy()
        out["predicted_total_score"] = np.round(scores, 2)
        out["predicted_grade"] = [labels[i] for i in pred_idx]
        out["predicted_grade_confidence"] = np.round(proba[np.arange(len(proba)), pred_idx], 3)
        for i, label in enumerate(labels):
            out[f"prob_grade_{label}"] = np.round(proba[:, i], 4)
        return out

    def predict_one(self, study_hours: float, attendance: float,
                    participation: float) -> Dict[str, Any]:
        df = pd.DataFrame([{"weekly_self_study_hours": study_hours,
                            "attendance_percentage": attendance,
                            "class_participation": participation}])
        row = self.predict_frame(df).iloc[0]
        return {
            "inputs": {"weekly_self_study_hours": float(study_hours),
                       "attendance_percentage": float(attendance),
                       "class_participation": float(participation)},
            "predicted_total_score": float(row["predicted_total_score"]),
            "predicted_grade": str(row["predicted_grade"]),
            "grade_confidence": float(row["predicted_grade_confidence"]),
            "grade_probabilities": {label: float(row[f"prob_grade_{label}"])
                                    for label in self.grade["labels"]},
            "model": self.reg["name"],
            "note": ("model prediction from study-behaviour features only; not a guarantee of "
                     "the student's actual result"),
        }

    # -- secondary exams model ---------------------------------------------
    def predict_exams(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if self.exams is None:
            raise FileNotFoundError("exams model not trained")
        feature_cols = self.exams["features"]
        missing = [c for c in feature_cols if c not in record]
        if missing:
            raise ValueError(f"missing exams features: {missing}")
        row = pd.DataFrame([{c: str(record[c]) for c in feature_cols}])
        X = self.exams["encoder"].transform(row)
        score = float(np.clip(self.exams["model"].predict(X)[0], 0, 100))
        cut = self.exams["category_cutoffs"]
        category = ("needs improvement" if score < cut["needs improvement"]
                    else "satisfactory" if score < cut["satisfactory"] else "good")
        return {"inputs": {c: str(record[c]) for c in feature_cols},
                "predicted_math_score": round(score, 2),
                "predicted_category": category,
                "model": self.exams["name"],
                "note": "trained on 700 exam records; coarse secondary reference model"}


def parse_student_json(payload: str) -> Dict[str, float]:
    data = json.loads(payload)
    mapping = {
        "study_hours": "weekly_self_study_hours", "hours": "weekly_self_study_hours",
        "weekly_self_study_hours": "weekly_self_study_hours",
        "attendance": "attendance_percentage", "attendance_percentage": "attendance_percentage",
        "participation": "class_participation", "class_participation": "class_participation",
    }
    out: Dict[str, float] = {}
    for key, value in data.items():
        if key in mapping:
            out[mapping[key]] = float(value)
    missing = [c for c in RAW_FEATURES if c not in out]
    if missing:
        raise ValueError(f"missing fields in --json: {missing}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict student performance.")
    parser.add_argument("--study-hours", type=float)
    parser.add_argument("--attendance", type=float)
    parser.add_argument("--participation", type=float)
    parser.add_argument("--json", help="JSON object with the three student fields")
    parser.add_argument("--input", help="CSV or parquet of students to score")
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "inference_predictions.csv"))
    parser.add_argument("--sample", type=int, default=0, help="score N rows of the test split")
    parser.add_argument("--exams-json", help="JSON object with the five exams features")
    args = parser.parse_args()

    predictor = Predictor()
    made_output = False

    if args.exams_json:
        print(json.dumps(predictor.predict_exams(json.loads(args.exams_json)), indent=2))
        made_output = True

    if args.json:
        fields = parse_student_json(args.json)
        print(json.dumps(predictor.predict_one(
            fields["weekly_self_study_hours"], fields["attendance_percentage"],
            fields["class_participation"]), indent=2))
        made_output = True
    elif args.study_hours is not None:
        if args.attendance is None or args.participation is None:
            parser.error("--study-hours requires --attendance and --participation")
        print(json.dumps(predictor.predict_one(args.study_hours, args.attendance,
                                               args.participation), indent=2))
        made_output = True

    if args.input:
        df = (pd.read_parquet(args.input) if args.input.lower().endswith(".parquet")
              else pd.read_csv(args.input))
        out = predictor.predict_frame(df)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"Scored {len(out):,} rows -> {args.out}")
        print(out.head(10).to_string(index=False))
        made_output = True

    if args.sample:
        processed = os.path.join(PROJECT_ROOT, "data", "processed", "tabular", "million_test.parquet")
        if not os.path.exists(processed):
            raise FileNotFoundError("processed test split missing; run `python src/train_model.py`")
        df = pd.read_parquet(processed)
        rows = df.sample(n=min(args.sample, len(df)), random_state=1337)
        out = predictor.predict_frame(rows)
        out.insert(0, "student_id", rows["student_id"].to_numpy())
        out["actual_total_score"] = rows["total_score"].to_numpy()
        out["actual_grade"] = rows["grade"].to_numpy()
        print(out.to_string(index=False))
        made_output = True

    if not made_output:
        print("Nothing to do. Try: python src/predict.py --study-hours 15 --attendance 85 "
              "--participation 6.5")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
