"""Natural-language demonstration over the trained student-performance models.

    python src/nl_demo.py --question "Which features have the greatest influence on predictions?"
    python src/nl_demo.py --question "Predict the mathematics score for a student with the provided characteristics" --context '{"study_hours": 18, "attendance": 92, "participation": 8}'
    python src/nl_demo.py --interactive

IMPORTANT — honest scope: this is NOT a trained language model. It is a small
template/retrieval layer that (a) computes factual statistics straight from the
datasets and (b) calls the trained tabular models for predictions. Every answer is
tagged with its evidence type so model output is never presented as fact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Optional

import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict import Predictor  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "tabular")

OBSERVATION = "OBSERVATION (computed from the dataset)"
PREDICTION = "PREDICTION (trained model estimate, not a guarantee)"
INTERPRETATION = "MODEL INTERPRETATION (measured on the trained model)"


class StudentPerformanceAssistant:
    def __init__(self) -> None:
        self.predictor = Predictor()
        metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
        self.metrics: Dict[str, Any] = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as fh:
                self.metrics = json.load(fh)
        self._exams_stats: Optional[Dict[str, Any]] = None
        self._m = pd.read_parquet(os.path.join(DATA_DIR, "million_test.parquet")) \
            if os.path.exists(os.path.join(DATA_DIR, "million_test.parquet")) else None

    # -- factual statistics ------------------------------------------------
    def exams_prep_stats(self) -> Dict[str, Any]:
        if self._exams_stats is not None:
            return self._exams_stats
        path = os.path.join(DATA_DIR, "exams_train.parquet")
        parts = [pd.read_parquet(os.path.join(DATA_DIR, f"exams_{s}.parquet"))
                 for s in ("train", "validation", "test")
                 if os.path.exists(os.path.join(DATA_DIR, f"exams_{s}.parquet"))]
        if not parts:
            self._exams_stats = {}
            return self._exams_stats
        exams = pd.concat(parts, ignore_index=True)
        grouped = exams.groupby("test preparation course")[
            ["math score", "reading score", "writing score"]].mean().round(2)
        counts = exams["test preparation course"].value_counts().to_dict()
        self._exams_stats = {"means": grouped.to_dict(orient="index"),
                             "counts": {str(k): int(v) for k, v in counts.items()},
                             "n": int(len(exams))}
        return self._exams_stats

    # -- parsers -----------------------------------------------------------
    @staticmethod
    def parse_from_text(text: str) -> Dict[str, float]:
        found: Dict[str, float] = {}
        hours = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:of\s*)?(?:study|studying|self[- ]?study)", text)
        if not hours:
            hours = re.search(r"stud(?:y|ies)\w*\s*(?:for\s*)?(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)", text)
        if hours:
            found["weekly_self_study_hours"] = float(hours.group(1))
        attendance = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", text)
        if attendance:
            found["attendance_percentage"] = float(attendance.group(1))
        participation = re.search(r"participation\s*(?:score|of|is|=|:)?\s*(\d+(?:\.\d+)?)", text)
        if not participation:
            participation = re.search(r"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*10", text)
        if participation:
            found["class_participation"] = float(participation.group(1))
        return found

    # -- answers -----------------------------------------------------------
    def answer(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        q = question.lower().strip()
        context = dict(context or {})

        if any(w in q for w in ("influence", "factor", "important", "affect", "drive", "impact")):
            return self._influence_answer(question)
        if any(w in q for w in ("test prep", "preparation", "prep course")):
            return self._prep_answer(question)
        if any(w in q for w in ("math score", "mathematics score", "maths score")):
            return self._exams_answer(question, context)
        if any(w in q for w in ("predict", "what score", "expected", "grade for", "performance of",
                                "estimate")):
            return self._prediction_answer(question, context)
        if any(w in q for w in ("how many", "dataset", "records", "describe", "what data")):
            return self._dataset_answer(question)
        return self._overview_answer(question)

    def _influence_answer(self, question: str) -> Dict[str, Any]:
        imp = self.metrics.get("feature_importance_permutation_rmse", {})
        ordered = sorted(imp.items(), key=lambda kv: -kv[1]) if imp else []
        lines = [INTERPRETATION,
                 "Permutation importance measured on the held-out test split "
                 "(increase in RMSE when a feature is shuffled):"]
        lines += [f"  {i+1}. {name}: {value:.4f} points" for i, (name, value) in enumerate(ordered)]
        if self._m is not None:
            corr = self._m[["weekly_self_study_hours", "attendance_percentage",
                            "class_participation", "total_score"]].corr()["total_score"]
            lines += ["", OBSERVATION, "Pearson correlation with total_score in the test split:"]
            lines += [f"  {k}: {v:+.3f}" for k, v in corr.items() if k != "total_score"]
            lines += ["", "Weekly self-study hours dominate; attendance and class participation "
                          "carry almost no linear signal in this dataset."]
        return {"question": question, "answer": "\n".join(lines),
                "evidence": "model_interpretation + dataset_observation"}

    def _prep_answer(self, question: str) -> Dict[str, Any]:
        stats = self.exams_prep_stats()
        if not stats:
            return {"question": question, "answer": "The exams dataset is not processed yet; run "
                                                    "`python src/train_model.py`.",
                    "evidence": "unavailable"}
        means = stats["means"]
        lines = [OBSERVATION,
                 f"Mean exam scores by test-preparation status (exams dataset, n={stats['n']}):",
                 f"{'group':<12}{'math':>8}{'reading':>9}{'writing':>9}{'rows':>7}"]
        for group, scores in means.items():
            rows = stats["counts"].get(group, 0)
            lines.append(f"{group:<12}{scores['math score']:>8.2f}{scores['reading score']:>9.2f}"
                         f"{scores['writing score']:>9.2f}{rows:>7}")
        lines += ["", "This is a descriptive difference between groups, not a causal effect: "
                      "students who chose to complete the course may already differ in other ways."]
        return {"question": question, "answer": "\n".join(lines), "evidence": "dataset_observation"}

    def _exams_answer(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        required = ["gender", "race/ethnicity", "parental level of education", "lunch",
                    "test preparation course"]
        if all(k in context for k in required):
            try:
                result = self.predictor.predict_exams(context)
                text = (f"{PREDICTION}\n{predictor_line(result)}\n"
                        f"Predicted category: {result['predicted_category']} "
                        f"(thresholds: <60 needs improvement, 60-79 satisfactory, >=80 good).")
                return {"question": question, "answer": text, "evidence": "prediction",
                        "details": result}
            except Exception as exc:
                return {"question": question, "answer": f"Could not score: {exc}",
                        "evidence": "error"}
        return {"question": question,
                "answer": ("To predict a mathematics score I need the five exams features "
                           "(gender, race/ethnicity, parental level of education, lunch, "
                           "test preparation course) via --context. The maths-score model is a "
                           "secondary model trained on only 700 exam records."),
                "evidence": "clarification_needed"}

    def _prediction_answer(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        fields = {"weekly_self_study_hours": None, "attendance_percentage": None,
                  "class_participation": None}
        alias = {"study_hours": "weekly_self_study_hours", "hours": "weekly_self_study_hours",
                 "attendance": "attendance_percentage", "participation": "class_participation",
                 **{k: k for k in fields}}
        for key, value in (context or {}).items():
            if key in alias:
                fields[alias[key]] = float(value)
        for key, value in self.parse_from_text(question).items():
            fields.setdefault(key, value)
            fields[key] = value if fields.get(key) is None else fields[key]
        if any(v is None for v in fields.values()):
            missing = [k for k, v in fields.items() if v is None]
            return {"question": question,
                    "answer": ("I need the student's study hours, attendance percentage and "
                               f"class participation to make a prediction (missing: {missing}). "
                               "You can also pass --context or use a sentence such as "
                               "\"predict the score for a student studying 18 hours with 92% "
                               "attendance and participation 8\"."),
                    "evidence": "clarification_needed"}
        result = self.predictor.predict_one(fields["weekly_self_study_hours"],
                                            fields["attendance_percentage"],
                                            fields["class_participation"])
        probs = result["grade_probabilities"]
        text = "\n".join([
            PREDICTION,
            f"Inputs: {result['inputs']}",
            f"Predicted total_score: {result['predicted_total_score']:.1f} / 100",
            f"Predicted grade: {result['predicted_grade']} "
            f"(confidence {result['grade_confidence'] * 100:.0f}%)",
            "Grade probabilities: " + ", ".join(f"{k}={v:.3f}" for k, v in probs.items()),
            f"Model: {result['model']}",
            "This is an estimate from study-behaviour features, not a guarantee of the outcome.",
        ])
        return {"question": question, "answer": text, "evidence": "prediction", "details": result}

    def _dataset_answer(self, question: str) -> Dict[str, Any]:
        splits = self.metrics.get("splits", {})
        tm = self.metrics.get("test_metrics", {})
        lines = [OBSERVATION,
                 "Primary dataset: Student Performance (1M records) — features weekly_self_study_hours, "
                 "attendance_percentage, class_participation; targets total_score and grade.",
                 f"Rows used: train {splits.get('train', 0):,}, validation "
                 f"{splits.get('validation', 0):,}, test {splits.get('test', 0):,}.",
                 "Secondary dataset: Students Performance in Exams (1,000 rows) — target math score; "
                 "trained separately because the schemas are incompatible."]
        if tm:
            lines.append(f"Measured test performance of the primary model: MAE {tm['mae']:.2f} "
                         f"points, R² {tm['r2']:.4f}.")
        return {"question": question, "answer": "\n".join(lines), "evidence": "dataset_observation"}

    def _overview_answer(self, question: str) -> Dict[str, Any]:
        lines = [OBSERVATION,
                 "I can answer questions about dataset statistics, feature influence, the "
                 "relationship between test preparation and exam scores, and model predictions "
                 "for a student's study hours, attendance and participation.",
                 "Example questions:",
                 "  - What factors influence student performance?",
                 "  - How does test preparation relate to exam performance?",
                 "  - Predict the score for a student studying 18 hours with 92% attendance and "
                 "participation 8.",
                 "  - Which features have the greatest influence on predictions?"]
        return {"question": question, "answer": "\n".join(lines), "evidence": "help"}


def predictor_line(result: Dict[str, Any]) -> str:
    return (f"Predicted math score: {result['predicted_math_score']} / 100 "
            f"(model: {result['model']}).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Natural-language demo over trained models.")
    parser.add_argument("--question")
    parser.add_argument("--context", help="JSON object with student characteristics")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--json-out", help="write the answer object to this JSON file")
    args = parser.parse_args()

    assistant = StudentPerformanceAssistant()
    context = json.loads(args.context) if args.context else None

    if args.interactive:
        print("Student Performance assistant — type 'exit' to stop.")
        print("Not a language model: answers are computed from the data or the trained model.\n")
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question or question.lower() in {"exit", "quit"}:
                break
            result = assistant.answer(question)
            print(f"\n{result['answer']}\n[{result['evidence']}]\n")
        return 0

    question = args.question or "What can you do?"
    result = assistant.answer(question, context)
    print(result["answer"])
    print(f"\n[{result['evidence']}]")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
