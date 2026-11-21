"""End-to-end tests for the Student Performance Intelligence System.

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v

Covers: saved artifacts load, predictions are reproducible and bounded, invalid input is
rejected, metrics in results/ match the model config, the insight engine's what-if math,
the HTTP API (predict / ask / meta / invalid payloads), and the Ollama layer's
availability handling and grounding guarantee.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
APP_DIR = os.path.join(PROJECT_ROOT, "app")
for path in (SRC_DIR, APP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import insight  # noqa: E402
import ollama_service  # noqa: E402
import server as app_server  # noqa: E402
from predict import Predictor  # noqa: E402


def _request(url: str, method: str = "GET", payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


class ArtifactTests(unittest.TestCase):
    def test_required_artifacts_exist(self):
        required = [
            "models/student_performance/best_model/model.joblib",
            "models/student_performance/best_model/grade_model.joblib",
            "models/student_performance/best_model/exams_model.joblib",
            "models/student_performance/preprocessor/million_preprocessor.joblib",
            "models/student_performance/preprocessor/feature_names.json",
            "models/student_performance/model_config.json",
            "results/metrics.json",
            "results/predictions.csv",
            "training_summary.md",
            "models/ollama/Modelfile.student-performance",
        ]
        for relative in required:
            self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, relative)), relative)

    def test_metrics_match_model_config(self):
        with open(os.path.join(PROJECT_ROOT, "results", "metrics.json"), encoding="utf-8") as fh:
            metrics = json.load(fh)
        with open(os.path.join(PROJECT_ROOT, "models", "student_performance",
                               "model_config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
        self.assertAlmostEqual(metrics["test_metrics"]["mae"], config["test_metrics"]["mae"],
                               places=6)
        self.assertAlmostEqual(metrics["test_metrics"]["r2"], config["test_metrics"]["r2"],
                               places=6)
        self.assertEqual(metrics["test_metrics"]["n_samples"], 150_000)


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor()

    def test_single_prediction_is_reproducible_and_bounded(self):
        first = self.predictor.predict_one(18, 92, 8)
        second = self.predictor.predict_one(18, 92, 8)
        self.assertEqual(first["predicted_total_score"], second["predicted_total_score"])
        self.assertTrue(0 <= first["predicted_total_score"] <= 100)
        self.assertIn(first["predicted_grade"], {"A", "B", "C", "D", "F"})
        self.assertAlmostEqual(sum(first["grade_probabilities"].values()), 1.0, places=2)

    def test_monotone_in_study_hours(self):
        low = self.predictor.predict_one(5, 85, 6.5)["predicted_total_score"]
        high = self.predictor.predict_one(30, 85, 6.5)["predicted_total_score"]
        self.assertGreater(high, low)

    def test_batch_matches_single(self):
        import pandas as pd

        df = pd.DataFrame([
            {"weekly_self_study_hours": 15.0, "attendance_percentage": 85.0,
             "class_participation": 6.5},
            {"weekly_self_study_hours": 30.0, "attendance_percentage": 50.0,
             "class_participation": 9.0},
        ])
        batch = self.predictor.predict_frame(df)
        single = self.predictor.predict_one(30, 50, 9)
        self.assertAlmostEqual(float(batch.iloc[1]["predicted_total_score"]),
                               single["predicted_total_score"], places=6)

    def test_invalid_rows_raise(self):
        import pandas as pd

        with self.assertRaises(ValueError):
            self.predictor.predict_frame(pd.DataFrame([
                {"weekly_self_study_hours": "abc", "attendance_percentage": 80,
                 "class_participation": 5}]))

    def test_exams_prediction(self):
        result = self.predictor.predict_exams({
            "gender": "female", "race/ethnicity": "group C",
            "parental level of education": "some college", "lunch": "standard",
            "test preparation course": "completed"})
        self.assertTrue(0 <= result["predicted_math_score"] <= 100)
        self.assertIn(result["predicted_category"],
                      {"needs improvement", "satisfactory", "good"})


class InsightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = insight.InsightEngine()

    def test_prediction_contains_grounded_fields(self):
        result = self.engine.predict(15, 85, 6.5)
        self.assertEqual(result["predicted_total_score"], 86.97)
        self.assertEqual(result["predicted_grade"], "A")
        self.assertEqual(result["dominant_feature"], "weekly_self_study_hours")
        self.assertTrue(result["drivers"][0]["importance_rmse_points"] > 1.0)
        self.assertAlmostEqual(result["expected_range"][0],
                               result["predicted_total_score"] - 6.119, places=1)

    def test_input_validation(self):
        for args in ((-1, 85, 6.5), (15, 120, 6.5), (15, 85, 12), ("x", 85, 6.5)):
            with self.assertRaises(ValueError, msg=str(args)):
                self.engine.predict(*args)

    def test_what_if_uses_model(self):
        suggestions = self.engine.suggestions(10, 80, 5)
        hours = [s for s in suggestions if s["lever"] == "weekly_self_study_hours"]
        self.assertGreaterEqual(len(hours), 2)
        base = self.engine.predictor.predict_one(10, 80, 5)["predicted_total_score"]
        for s in hours:
            self.assertAlmostEqual(s["predicted_change"],
                                   s["predicted_score"] - base, places=1)
            self.assertGreater(s["to"], s["from"])
        attendance = [s for s in suggestions if s["lever"] == "attendance_percentage"][0]
        self.assertLess(abs(attendance["predicted_change"]), 1.0,
                        "dataset shows no attendance signal; the model must not invent one")

    def test_answers_are_grounded_or_clarifying(self):
        stats = self.engine.answer("What is the average student score?")
        self.assertEqual(stats["evidence"], "dataset_observation")
        self.assertIn("mean total score", stats["answer"])

        importance = self.engine.answer("Which features are most important?")
        self.assertIn("weekly_self_study_hours", importance["answer"])

        clarify = self.engine.answer("predict the score")
        self.assertEqual(clarify["evidence"], "clarification_needed")

        parsed = self.engine.answer(
            "predict the score for a student studying 18 hours with 92% attendance "
            "and participation 8")
        self.assertEqual(parsed["evidence"], "prediction")
        self.assertEqual(parsed["details"]["inputs"]["weekly_self_study_hours"], 18.0)

        limits = self.engine.answer("What are the limitations of this prediction system?")
        self.assertIn("right-censored", limits["answer"])

    def test_dataset_statistics_match_predictions_csv(self):
        import pandas as pd

        stats = self.engine.dataset_statistics()
        frame = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "predictions.csv"),
                            usecols=["actual_total_score"])
        self.assertAlmostEqual(stats["mean_score"],
                               round(float(frame["actual_total_score"].mean()), 3), places=2)


class HttpApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.application = app_server.build_server(0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_index_and_static_assets(self):
        status, body = _request(self.base + "/")
        self.assertEqual(status, 200)
        self.assertIn("Student Performance Intelligence System", body)
        for asset, needle in (("/styles.css", "--accent"), ("/app.js", "renderPrediction")):
            status, body = _request(self.base + asset)
            self.assertEqual(status, 200)
            self.assertIn(needle, body)

    def test_meta_endpoint(self):
        status, body = _request(self.base + "/api/meta")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["metrics"]["test_metrics"]["mae"], 6.119)
        self.assertGreaterEqual(len(data["importance"]), 6)
        self.assertTrue(data["dataset"]["available"])

    def test_health_endpoint(self):
        status, body = _request(self.base + "/api/health")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["status"], "ok")
        self.assertIn("ollama", data)

    def test_predict_endpoint(self):
        status, body = _request(self.base + "/api/predict", "POST",
                                {"study_hours": 15, "attendance": 85, "participation": 6.5})
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["predicted_total_score"], 86.97)
        self.assertEqual(data["predicted_grade"], "A")

    def test_predict_rejects_bad_payloads(self):
        status, body = _request(self.base + "/api/predict", "POST",
                                {"study_hours": 15, "attendance": 85})
        self.assertEqual(status, 400)
        self.assertIn("required", body)
        status, _ = _request(self.base + "/api/predict", "POST",
                             {"study_hours": 500, "attendance": 85, "participation": 6.5})
        self.assertEqual(status, 400)

    def test_ask_endpoint_falls_back_without_ollama(self):
        status, body = _request(self.base + "/api/ask", "POST",
                                {"question": "Which features are most important?"})
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertIn(data["source"], {"local-deterministic", "ollama+tools",
                                       "ollama+explanation", "ollama"})
        # The LLM may humanise the column name; the deterministic facts keep it verbatim.
        self.assertIn("self-study hours", data["answer"].lower())
        self.assertIn("weekly_self_study_hours", data["deterministic_answer"])

    def test_ask_uses_llm_when_available(self):
        status, _ = _request(self.base + "/api/health")
        health = json.loads(_request(self.base + "/api/health")[1])
        if not (health["ollama"]["available"] and health["ollama"]["model"]):
            self.skipTest("Ollama not available")
        status, body = _request(self.base + "/api/ask", "POST",
                                {"question": "What is the average student score?"})
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertIn(data["source"], {"ollama+explanation", "ollama"})
        self.assertIn("84.29", data["answer"])
        self.assertEqual(data["audit"], [])

    def test_ask_requires_question(self):
        status, _ = _request(self.base + "/api/ask", "POST", {"question": "  "})
        self.assertEqual(status, 400)

    def test_unknown_route(self):
        status, _ = _request(self.base + "/api/nope")
        self.assertEqual(status, 404)
        status, _ = _request(self.base + "/../README.md")
        self.assertIn(status, (403, 404))


class OllamaTests(unittest.TestCase):
    def test_status_is_structured(self):
        service = ollama_service.OllamaService()
        status = service.status(refresh=True)
        self.assertIn("available", status)
        self.assertIn("models", status)
        if not status["available"]:
            self.assertIsNone(status["model"])
            self.assertTrue(status["error"])

    def test_fallback_answer_never_invents_numbers(self):
        service = ollama_service.OllamaService()
        result = service.answer("Predict the score for 18 hours, 92% attendance, "
                                "participation 8", context=None)
        self.assertIn("answer", result)
        if result["source"] == "local-deterministic":
            self.assertIn("18.0", json.dumps(result["details"], default=str))

    def test_tool_implementations_call_the_real_model(self):
        service = ollama_service.OllamaService()
        direct = service.engine.predict(18, 92, 8)["predicted_total_score"]
        via_tool = service.call_tool("predict_student_performance",
                                     {"study_hours": 18, "attendance": 92,
                                      "participation": 8})
        self.assertEqual(via_tool["predicted_total_score"], direct)
        error = service.call_tool("predict_student_performance",
                                  {"study_hours": 999, "attendance": 92, "participation": 8})
        self.assertIn("error", error)

    def test_numeric_audit_flags_invented_values(self):
        facts = {"answer": "score 86.97 grade A", "details": {}}
        flags = ollama_service.OllamaService._numeric_audit(
            "The student will score 86.97 and 99.5.", facts)
        self.assertIn("99.5", flags)
        self.assertNotIn("86.97", flags)


if __name__ == "__main__":
    unittest.main(verbosity=2)
