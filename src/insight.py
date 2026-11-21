"""Grounded student-performance intelligence layer.

This module is the single, deterministic source of truth used by the web
application and by the Ollama bridge.  It never invents numbers: every value it
returns is either

* read from the measured artifacts in ``results/`` (metrics, permutation
  importance, inspection report), or
* read from the processed dataset splits, or
* produced by the trained joblib models loaded through :class:`Predictor`.

The natural-language model (Ollama) is only allowed to *explain* these facts;
it is never asked to compute a score.

    python src/insight.py --predict 15 85 6.5
    python src/insight.py --stats
    python src/insight.py --ask "What factors influence student performance?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict import Predictor  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "tabular")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "student_performance")

OBSERVATION = "OBSERVATION (computed from the dataset)"
PREDICTION = "PREDICTION (trained model estimate, not a guarantee)"
INTERPRETATION = "MODEL INTERPRETATION (measured on the trained model)"

# Documented input domains of the 1M training table.
INPUT_BOUNDS = {
    "weekly_self_study_hours": (0.0, 40.0),
    "attendance_percentage": (0.0, 100.0),
    "class_participation": (0.0, 10.0),
}

# Score bands used only for wording; A/B/C/D/F cut-offs are the dataset's own.
BANDS = (
    (85.0, "strong", "The model places this student in the A band, where scores concentrate "
                     "at the top of the scale."),
    (70.0, "solid", "The model places this student in the B band: comfortably above the "
                    "class average, with room to reach A."),
    (55.0, "developing", "The model places this student in the C band, below the dataset "
                         "average."),
    (40.0, "at risk", "The model places this student in the D band, well below the dataset "
                      "average."),
    (0.0, "critical", "The model places this student in the F band."),
)


def _finite(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


class InsightEngine:
    """Deterministic predictions, statistics and explanations."""

    def __init__(self) -> None:
        self.predictor = Predictor()
        self.metrics: Dict[str, Any] = self._load_json("metrics.json")
        self.model_config: Dict[str, Any] = self._load_json(
            os.path.join("..", "models", "student_performance", "model_config.json"))
        self._test_frame: Optional[pd.DataFrame] = None

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _load_json(*relative: str) -> Dict[str, Any]:
        path = os.path.join(RESULTS_DIR, *relative)
        path = os.path.normpath(path)
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    @property
    def test_frame(self) -> Optional[pd.DataFrame]:
        if self._test_frame is None:
            path = os.path.join(DATA_DIR, "million_test.parquet")
            if os.path.exists(path):
                self._test_frame = pd.read_parquet(path)
        return self._test_frame

    @staticmethod
    def validate(hours: Any, attendance: Any, participation: Any) -> Dict[str, float]:
        values = {
            "weekly_self_study_hours": _finite(hours, "study hours"),
            "attendance_percentage": _finite(attendance, "attendance"),
            "class_participation": _finite(participation, "class participation"),
        }
        for name, (low, high) in INPUT_BOUNDS.items():
            if not low <= values[name] <= high:
                raise ValueError(f"{name} must be between {low:g} and {high:g}; "
                                 f"got {values[name]:g}")
        return values

    # -- model facts -------------------------------------------------------
    @staticmethod
    def _round(value: Any, digits: int = 4) -> Optional[float]:
        """JSON-safe rounding: missing/non-finite values become None, never NaN."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(number):
            return None
        return round(number, digits)

    def model_metrics(self) -> Dict[str, Any]:
        test = self.metrics.get("test_metrics", {}) or {}
        grade = (self.metrics.get("grade_classifier", {}) or {}).get("test_metrics", {}) or {}
        exams = (self.metrics.get("exams_model", {}) or {}).get("test_metrics", {}) or {}
        exams_block = self.metrics.get("exams_model", {}) or {}
        return {
            "primary_model": self.model_config.get("name", "hist_gradient_boosting"),
            "model_family": self.model_config.get("family", "hist_gradient_boosting"),
            "features": self.model_config.get("features", []),
            "target": self.model_config.get("target", "total_score"),
            "trained_on_rows": int(self.model_config.get("n_train_used", 0)),
            "test_metrics": {
                "mae": self._round(test.get("mae")),
                "rmse": self._round(test.get("rmse")),
                "r2": self._round(test.get("r2")),
                "n_samples": int(test.get("n_samples") or 0),
            },
            "grade_model": {
                "accuracy": self._round(grade.get("accuracy")),
                "f1_weighted": self._round(grade.get("f1_weighted")),
                "roc_auc_ovr_macro": self._round(grade.get("roc_auc_ovr_macro")),
            },
            "exams_model": {
                "name": exams_block.get("selected", "linear_regression"),
                "mae": self._round(exams.get("mae")),
                "r2": self._round(exams.get("r2")),
                "category_accuracy": self._round(exams.get("category_accuracy")),
            },
            "artifacts_dir": os.path.relpath(MODELS_DIR, PROJECT_ROOT),
        }

    def feature_importance(self) -> List[Dict[str, Any]]:
        imp = self.metrics.get("feature_importance_permutation_rmse", {}) or {}
        rows = [{"feature": name, "importance_rmse_points": round(float(value), 6)}
                for name, value in imp.items()]
        rows.sort(key=lambda r: -abs(r["importance_rmse_points"]))
        return rows

    def dataset_statistics(self) -> Dict[str, Any]:
        frame = self.test_frame
        if frame is None:
            return {"available": False,
                    "reason": "processed test split missing; run `python src/train_model.py`"}
        scores = frame["total_score"]
        corr = frame[["weekly_self_study_hours", "attendance_percentage",
                      "class_participation", "total_score"]].corr()["total_score"]
        grades = frame["grade"].value_counts().sort_index().to_dict()
        splits = self.metrics.get("splits", {})
        return {
            "available": True,
            "split": "test",
            "rows": int(len(frame)),
            "splits": splits,
            "mean_score": round(float(scores.mean()), 3),
            "median_score": round(float(scores.median()), 3),
            "std_score": round(float(scores.std()), 3),
            "min_score": round(float(scores.min()), 3),
            "max_score": round(float(scores.max()), 3),
            "share_scoring_100": round(float((scores >= 100).mean()), 4),
            "grade_counts": {str(k): int(v) for k, v in grades.items()},
            "correlation_with_score": {k: round(float(v), 4) for k, v in corr.items()
                                       if k != "total_score"},
        }

    # -- prediction --------------------------------------------------------
    def predict(self, study_hours: float, attendance: float,
                participation: float) -> Dict[str, Any]:
        values = self.validate(study_hours, attendance, participation)
        result = self.predictor.predict_one(values["weekly_self_study_hours"],
                                            values["attendance_percentage"],
                                            values["class_participation"])
        score = result["predicted_total_score"]
        mae = float(self.metrics.get("test_metrics", {}).get("mae", 6.12))
        band, band_text = next(((name, text) for cut, name, text in BANDS if score >= cut),
                               ("critical", BANDS[-1][2]))
        importance = self.feature_importance()
        dominant = importance[0]["feature"] if importance else "weekly_self_study_hours"

        drivers = []
        for row in importance:
            effect = row["importance_rmse_points"]
            if effect >= 0.5:
                level = "dominant"
            elif effect >= 0.05:
                level = "moderate"
            else:
                level = "negligible"
            drivers.append({**row, "level": level})

        return {
            "inputs": values,
            "predicted_total_score": round(float(score), 2),
            "expected_range": [round(max(0.0, score - mae), 2),
                               round(min(100.0, score + mae), 2)],
            "predicted_grade": result["predicted_grade"],
            "grade_confidence": result["grade_confidence"],
            "grade_probabilities": result["grade_probabilities"],
            "performance_band": band,
            "interpretation": (f"{band_text} The model's typical error on unseen data is "
                               f"{mae:.2f} score points (test MAE), so a score near "
                               f"{score:.1f} is expected to fall within roughly "
                               f"{max(0.0, score - mae):.1f}-{min(100.0, score + mae):.1f}."),
            "dominant_feature": dominant,
            "drivers": drivers,
            "suggestions": self.suggestions(values["weekly_self_study_hours"],
                                            values["attendance_percentage"],
                                            values["class_participation"]),
            "model": result["model"],
            "evidence": "prediction + dataset_observation + model_interpretation",
        }

    def suggestions(self, study_hours: float, attendance: float,
                    participation: float) -> List[Dict[str, Any]]:
        """Model-based what-if estimates (not causal claims)."""
        base = self.predictor.predict_one(study_hours, attendance,
                                          participation)["predicted_total_score"]
        out: List[Dict[str, Any]] = []
        for delta in (2.0, 4.0, 8.0):
            new_hours = min(INPUT_BOUNDS["weekly_self_study_hours"][1], study_hours + delta)
            if new_hours <= study_hours:
                break
            new_score = self.predictor.predict_one(new_hours, attendance,
                                                   participation)["predicted_total_score"]
            out.append({
                "lever": "weekly_self_study_hours",
                "from": round(study_hours, 2),
                "to": round(new_hours, 2),
                "predicted_score": round(float(new_score), 2),
                "predicted_change": round(float(new_score - base), 2),
                "basis": "model what-if (other inputs held fixed)",
            })
        new_attendance = min(100.0, attendance + 10.0)
        if new_attendance > attendance:
            new_score = self.predictor.predict_one(study_hours, new_attendance,
                                                   participation)["predicted_total_score"]
            out.append({
                "lever": "attendance_percentage",
                "from": round(attendance, 2),
                "to": round(new_attendance, 2),
                "predicted_score": round(float(new_score), 2),
                "predicted_change": round(float(new_score - base), 2),
                "basis": "model what-if (other inputs held fixed)",
            })
        return out

    # -- question answering ------------------------------------------------
    def limitations(self) -> List[str]:
        stats = self.dataset_statistics()
        return [
            "The model was trained on three study-behaviour features only; it does not know "
            "the subject, teacher, prior attainment, health or home circumstances.",
            f"Typical error on unseen students is about "
            f"{self.metrics.get('test_metrics', {}).get('mae', 6.12):.2f} score points (MAE); "
            f"the model explains about "
            f"{100 * self.metrics.get('test_metrics', {}).get('r2', 0.7175):.1f}% of the "
            f"variance in the held-out split (R²).",
            "26.8% of the training scores are exactly 100, so the target is right-censored and "
            "R² saturates near 0.72; the model cannot distinguish a 100 from a 100.",
            "Grade is almost a deterministic binning of the score (99.79% agreement), so the "
            "grade classifier trained on behaviour alone is only moderately accurate.",
            "Attendance and class participation have near-zero measured signal in this "
            "dataset; the model does not establish that changing them changes marks.",
            "Predictions are statistical associations for a synthetic-looking dataset, not "
            "causal effects and not a guarantee for any individual student.",
        ]

    def answer(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Deterministic answer used as the grounding facts for the LLM."""
        q = (question or "").lower().strip()
        ctx = dict(context or {})

        aliases = {"study_hours": "weekly_self_study_hours", "hours": "weekly_self_study_hours",
                   "attendance": "attendance_percentage",
                   "participation": "class_participation"}
        resolved = {}
        for key, value in ctx.items():
            resolved[aliases.get(key, key)] = value
        numeric = self._numbers_from_text(question)

        wants_limits = any(w in q for w in ("limitation", "weakness", "caveat", "reliable",
                                            "accuracy", "accura", "trust"))
        # "prediction system" in a limitations question must not be read as a request to score.
        wants_prediction = (not wants_limits) and any(
            w in q for w in ("predict", "expected score", "expected grade", "what score",
                             "what grade", "estimate", "performance of", "score for"))
        wants_improve = any(w in q for w in ("improve", "better", "advice", "suggest",
                                             "what should", "how can"))
        wants_importance = any(w in q for w in ("important", "influence", "factor", "affect",
                                                "impact", "driver", "relationship", "relate"))
        wants_stats = any(w in q for w in ("average", "mean", "dataset", "records", "how many",
                                           "distribution", "describe"))

        if wants_prediction:
            # Numbers typed in the question win over the form values sent as context.
            fields = {k: numeric.get(k, resolved.get(k)) for k in INPUT_BOUNDS}
            missing = [k for k, v in fields.items() if v is None]
            if missing:
                return self._clarify(missing, question)
            result = self.predict(fields["weekly_self_study_hours"],
                                  fields["attendance_percentage"],
                                  fields["class_participation"])
            lines = [
                PREDICTION,
                f"Predicted total score: {result['predicted_total_score']:.1f} / 100 "
                f"(expected range {result['expected_range'][0]:.1f}-"
                f"{result['expected_range'][1]:.1f}).",
                f"Predicted grade: {result['predicted_grade']} "
                f"(confidence {result['grade_confidence'] * 100:.0f}%).",
                result["interpretation"],
                "",
                "Grade probabilities: " + ", ".join(
                    f"{k}={v:.3f}" for k, v in result["grade_probabilities"].items()),
            ]
            return {"answer": "\n".join(lines), "evidence": "prediction", "details": result}

        if wants_improve:
            fields = {k: numeric.get(k, resolved.get(k)) for k in INPUT_BOUNDS}
            if any(v is None for v in fields.values()):
                return self._clarify([k for k, v in fields.items() if v is None], question)
            result = self.predict(fields["weekly_self_study_hours"],
                                  fields["attendance_percentage"],
                                  fields["class_participation"])
            lines = [PREDICTION, "Model what-if estimates, holding the other inputs fixed:"]
            for s in result["suggestions"]:
                lines.append(f"  - {s['lever']}: {s['from']:.1f} -> {s['to']:.1f} gives "
                             f"{s['predicted_score']:.1f} "
                             f"({s['predicted_change']:+.2f} points).")
            lines += ["", "Important: these are model estimates for this dataset. Only weekly "
                          "self-study hours has meaningful measured signal; the attendance "
                          "change shows how little the model attributes to it. They are not "
                          "causal effects and not a plan guarantee."]
            return {"answer": "\n".join(lines), "evidence": "prediction",
                    "details": result}

        if wants_limits:
            lines = [INTERPRETATION, "Measured limitations of this system:"]
            lines += [f"  - {item}" for item in self.limitations()]
            return {"answer": "\n".join(lines), "evidence": "model_interpretation",
                    "details": {"limitations": self.limitations()}}

        if wants_importance:
            rows = self.feature_importance()
            lines = [
                INTERPRETATION,
                "Permutation importance on the held-out test split. Each value is how many "
                "points the model's RMSE ERROR increases when that feature is shuffled - it "
                "is an error increase, not a change in a student's marks:"]
            lines += [f"  {i + 1}. {r['feature']}: RMSE {r['importance_rmse_points']:+.4f} "
                      f"points when shuffled" for i, r in enumerate(rows)]
            stats = self.dataset_statistics()
            if stats.get("available"):
                lines += ["", OBSERVATION, "Pearson correlation with total_score (test split):"]
                lines += [f"  {k}: {v:+.3f}"
                          for k, v in stats["correlation_with_score"].items()]
                lines += ["", "Weekly self-study hours dominate. Attendance and class "
                              "participation carry almost no measured signal, so the model "
                              "does not support claims that they change marks."]
            return {"answer": "\n".join(lines), "evidence": "model_interpretation",
                    "details": {"importance": rows}}

        if wants_stats:
            stats = self.dataset_statistics()
            if not stats.get("available"):
                return {"answer": stats.get("reason", "dataset unavailable"),
                        "evidence": "unavailable", "details": stats}
            lines = [OBSERVATION,
                     f"Primary dataset (test split, n={stats['rows']:,}): mean total score "
                     f"{stats['mean_score']:.2f}, median {stats['median_score']:.2f}, "
                     f"SD {stats['std_score']:.2f}, range "
                     f"{stats['min_score']:.1f}-{stats['max_score']:.1f}.",
                     f"{stats['share_scoring_100'] * 100:.1f}% of these scores are exactly 100.",
                     "Grade counts: " + ", ".join(f"{k}={v:,}"
                                                  for k, v in stats["grade_counts"].items()),
                     "Secondary dataset: Students Performance in Exams, 1,000 rows "
                     "(700/150/150 split), target math score, trained separately."]
            tm = self.metrics.get("test_metrics", {})
            if tm:
                lines.append(f"Measured primary-model test performance: MAE {tm['mae']:.2f}, "
                             f"RMSE {tm['rmse']:.2f}, R² {tm['r2']:.4f}.")
            return {"answer": "\n".join(lines), "evidence": "dataset_observation",
                    "details": stats}

        lines = [
            OBSERVATION,
            "I can answer with measured data and trained-model output about:",
            "  - dataset size, average score and grade distribution",
            "  - which features influence predictions (permutation importance)",
            "  - a predicted score and grade for a student's study hours, attendance and "
            "participation",
            "  - model what-if estimates for improving study time",
            "  - the measured limitations and accuracy of the system",
            "Provide study hours, attendance percentage and participation (0-10) for a "
            "prediction, for example: \"predict the score for 18 hours, 92% attendance, "
            "participation 8\".",
        ]
        return {"answer": "\n".join(lines), "evidence": "help", "details": {}}

    @staticmethod
    def _numbers_from_text(text: str) -> Dict[str, float]:
        import re

        found: Dict[str, float] = {}
        hours = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)", text, re.I)
        if hours:
            found["weekly_self_study_hours"] = float(hours.group(1))
        attendance = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", text, re.I)
        if attendance:
            found["attendance_percentage"] = float(attendance.group(1))
        participation = re.search(
            r"participation\s*(?:score|of|is|=|:|at)?\s*(\d+(?:\.\d+)?)", text, re.I)
        if participation:
            found["class_participation"] = float(participation.group(1))
        return found

    @staticmethod
    def _clarify(missing: List[str], question: str) -> Dict[str, Any]:
        return {
            "answer": ("I need the student's weekly self-study hours, attendance percentage "
                       f"and class participation (0-10) to make a prediction. Missing: "
                       f"{', '.join(missing)}. Example: \"predict the score for 18 hours, "
                       "92% attendance, participation 8\"."),
            "evidence": "clarification_needed",
            "details": {"missing": missing, "question": question},
        }

    # -- tool schema payloads ---------------------------------------------
    def tool_facts(self) -> Dict[str, Any]:
        return {"metrics": self.model_metrics(), "importance": self.feature_importance(),
                "dataset": self.dataset_statistics(), "limitations": self.limitations()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Grounded student-performance insights.")
    parser.add_argument("--predict", nargs=3, type=float,
                        metavar=("STUDY_HOURS", "ATTENDANCE", "PARTICIPATION"))
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--ask")
    args = parser.parse_args()

    engine = InsightEngine()
    if args.predict:
        print(json.dumps(engine.predict(*args.predict), indent=2))
        return 0
    if args.stats:
        print(json.dumps(engine.dataset_statistics(), indent=2))
        return 0
    if args.metrics:
        print(json.dumps(engine.model_metrics(), indent=2))
        return 0
    if args.ask:
        print(json.dumps(engine.answer(args.ask), indent=2))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
