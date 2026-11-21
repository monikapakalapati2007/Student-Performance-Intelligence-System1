"""Ollama integration: the language model explains, the trained model computes.

Architecture (implemented here and in ``app/server.py``)::

    USER QUESTION
         |
         v
    STUDENT PERFORMANCE APPLICATION  (insight.InsightEngine)
         |  validates input, calls the joblib models, computes statistics
         v
    DETERMINISTIC ANSWER + FACTS (numbers, evidence type, tool payload)
         |
         v
    OLLAMA LANGUAGE MODEL  (llama-family, local, http://127.0.0.1:11434)
         |  rewrites the facts in plain language (default mode)
         |  or calls tools itself (optional --tools mode)
         v
    NUMERIC AUDIT  -> prose kept only if it adds no new numbers
         |
         v
    FINAL RESPONSE

Two modes:

* ``explain`` (default) — most reliable.  The engine has already produced the
  correct answer; the LLM rewrites it.  Any score-like number in the prose that
  is absent from the facts discards the prose and returns the deterministic text.
* ``tools`` (opt-in) — genuine tool calling.  Used for demonstration; a 1B model
  occasionally picks the wrong tool, and the numeric audit still guards the output.

If Ollama is not running the deterministic answer is returned with
``source = "local-deterministic"`` and an explicit note.  The app never breaks.

    python src/ollama_service.py --status
    python src/ollama_service.py --ask "What factors influence student performance?"
    python src/ollama_service.py --ask "Predict the score for 18 hours, 92% attendance, participation 8"
    python src/ollama_service.py --ask "..." --tools
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from insight import InsightEngine  # noqa: E402
from ollama_prompt import (ADVISOR_MODEL, EXPLAIN_SYSTEM_PROMPT,  # noqa: E402
                           SYSTEM_PROMPT)

DEFAULT_BASE = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if DEFAULT_BASE and not DEFAULT_BASE.startswith("http"):
    DEFAULT_BASE = "http://" + DEFAULT_BASE

PREFERRED_MODELS = (ADVISOR_MODEL, "llama3.2:1b", "llama3.2:3b", "qwen2.5:1.5b",
                    "qwen2.5:3b", "phi3.5:3.8b", "llama3.2")

MAX_TOOL_ROUNDS = 3

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "predict_student_performance",
            "description": ("Score one student with the trained HistGradientBoosting model. "
                            "Returns the predicted total score (0-100), expected range, "
                            "predicted letter grade with probabilities, and model-based "
                            "what-if suggestions."),
            "parameters": {
                "type": "object",
                "properties": {
                    "study_hours": {"type": "number",
                                    "description": "weekly self-study hours, 0-40"},
                    "attendance": {"type": "number",
                                   "description": "attendance percentage, 0-100"},
                    "participation": {"type": "number",
                                      "description": "class participation score, 0-10"},
                },
                "required": ["study_hours", "attendance", "participation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_feature_importance",
            "description": ("Permutation importance measured on the held-out test split: how "
                            "many RMSE points the model loses when each feature is shuffled."),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataset_statistics",
            "description": ("Measured statistics of the primary dataset's test split: rows, "
                            "mean/median/SD/range of total_score, share scoring 100, grade "
                            "counts, and correlations with total_score."),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_metrics",
            "description": ("Measured evaluation metrics of the trained models (regression "
                            "MAE/RMSE/R2, grade classifier accuracy/F1/AUC, exams model)."),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_improvement_suggestions",
            "description": ("Model what-if estimates for changing study hours and attendance "
                            "for one student, plus the measured limitations of the system."),
            "parameters": {
                "type": "object",
                "properties": {
                    "study_hours": {"type": "number"},
                    "attendance": {"type": "number"},
                    "participation": {"type": "number"},
                },
                "required": ["study_hours", "attendance", "participation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_limitations",
            "description": "Measured limitations and reliability notes of the prediction system.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class OllamaUnavailable(RuntimeError):
    pass


class OllamaService:
    def __init__(self, engine: Optional[InsightEngine] = None,
                 base_url: str = DEFAULT_BASE, model: Optional[str] = None) -> None:
        self.engine = engine or InsightEngine()
        self.base_url = base_url.rstrip("/")
        self._model = model or os.environ.get("OLLAMA_MODEL")
        self._status_cache: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None

    # -- HTTP --------------------------------------------------------------
    def _request(self, path: str, payload: Optional[Dict[str, Any]] = None,
                 timeout: float = 10.0) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise OllamaUnavailable(f"Ollama HTTP {exc.code}: {detail}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaUnavailable(f"cannot reach Ollama at {self.base_url}: {exc}") from None

    # -- discovery ---------------------------------------------------------
    def status(self, refresh: bool = False) -> Dict[str, Any]:
        if self._status_cache is not None and not refresh:
            return self._status_cache
        info: Dict[str, Any] = {"base_url": self.base_url, "available": False,
                                "models": [], "model": None, "error": None}
        try:
            tags = self._request("/api/tags", timeout=4.0)
            models = [m.get("name", "") for m in tags.get("models", []) if m.get("name")]
            info["available"] = True
            info["models"] = models
            info["model"] = self._choose_model(models)
            if not info["model"]:
                info["error"] = ("Ollama is running but has no model installed; run "
                                 "`python src/ollama_setup.py --pull --create`")
        except OllamaUnavailable as exc:
            info["error"] = str(exc)
        self._status_cache = info
        return info

    def _choose_model(self, models: List[str]) -> Optional[str]:
        if self._model:
            return self._model
        for preferred in PREFERRED_MODELS:
            for name in models:
                if name == preferred or name.split(":")[0] == preferred:
                    return name
        return models[0] if models else None

    # -- tool execution ----------------------------------------------------
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if name == "predict_student_performance":
                return self.engine.predict(float(arguments["study_hours"]),
                                           float(arguments["attendance"]),
                                           float(arguments["participation"]))
            if name == "get_improvement_suggestions":
                values = self.engine.validate(float(arguments["study_hours"]),
                                              float(arguments["attendance"]),
                                              float(arguments["participation"]))
                return {"suggestions": self.engine.suggestions(**values),
                        "limitations": self.engine.limitations()}
            if name == "get_feature_importance":
                return {"importance": self.engine.feature_importance(),
                        "dataset": self.engine.dataset_statistics()}
            if name == "get_dataset_statistics":
                return self.engine.dataset_statistics()
            if name == "get_model_metrics":
                return self.engine.model_metrics()
            if name == "get_system_limitations":
                return {"limitations": self.engine.limitations()}
            return {"error": f"unknown tool {name!r}"}
        except (ValueError, KeyError, TypeError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    # -- chat --------------------------------------------------------------
    def _chat(self, messages: List[Dict[str, Any]], model: str,
              tools: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.15, "num_ctx": 4096, "num_predict": 400},
        }
        if tools:
            payload["tools"] = TOOLS
        return self._request("/api/chat", payload, timeout=240.0)

    def _chat_safe(self, messages: List[Dict[str, Any]], model: str,
                   tools: bool) -> Dict[str, Any]:
        """Chat with one downgrade retry.

        llama3.2:1b occasionally emits a malformed tool-call grammar and llama.cpp
        answers HTTP 500.  The grounded facts are already in the prompt, so retrying
        without tools still produces a grounded answer.
        """
        if not tools:
            return self._chat(messages, model, tools=False)
        try:
            return self._chat(messages, model, tools=True)
        except OllamaUnavailable as exc:
            if "500" not in str(exc):
                raise
            self.last_error = str(exc)
            return self._chat(messages, model, tools=False)

    @staticmethod
    def _text(response: Dict[str, Any]) -> str:
        return OllamaService._clean_prose(
            (response.get("message") or {}).get("content") or "")

    @staticmethod
    def _clean_prose(text: str) -> str:
        """Remove the small model's habitual preamble lines."""
        import re

        lines = [line.rstrip() for line in text.strip().splitlines()]
        while lines and (not lines[0].strip() or re.match(
                r"(?i)^(here('| i)s|here are|based on the|the (application|model)|rewritten|"
                r"sure|okay|below is)",
                lines[0].strip()) and lines[0].strip().endswith(":")):
            lines.pop(0)
        return "\n".join(lines).strip()

    # -- modes -------------------------------------------------------------
    def _explain(self, model: str, question: str, facts: Dict[str, Any],
                 previous: Optional[str] = None,
                 problems: Optional[List[str]] = None) -> str:
        """Default mode: rewrite the authoritative answer without changing numbers."""
        request = (
            "AUTHORITATIVE ANSWER FROM THE APPLICATION (do not change any number):\n"
            f"{facts['answer']}\n\n"
            f"QUESTION: {question}\n\n"
            "Rewrite the authoritative answer for the student in plain language, "
            "keeping every number exactly as given. Do not add numbers, do not call "
            "tools, and do not use code blocks.")
        if previous and problems:
            request += ("\n\nYour previous draft was rejected for these reasons: "
                        f"{'; '.join(problems)}. Rejected draft:\n{previous}\n\n"
                        "Write a corrected version now. Remove every rejected number and "
                        "every causal word. Describe measured associations only, and only "
                        "with the wording and numbers used above.")
        messages = [
            {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
            {"role": "user", "content": request},
        ]
        return self._text(self._chat(messages, model, tools=False))

    def _tool_loop(self, model: str, question: str,
                   context: Optional[Dict[str, Any]], facts: Dict[str, Any],
                   result: Dict[str, Any]) -> str:
        """Optional mode: the model chooses tools; results are computed locally."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(facts)},
            {"role": "user", "content": self._user_prompt(question, context)},
        ]
        seen: set = set()
        prose = ""
        for _round in range(MAX_TOOL_ROUNDS):
            response = self._chat_safe(messages, model, tools=True)
            message = response.get("message", {}) or {}
            calls = message.get("tool_calls") or []
            text = (message.get("content") or "").strip()
            if not calls:
                return text

            fresh = []
            for call in calls:
                fn = call.get("function", {}) or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments")
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args or "{}")
                    except json.JSONDecodeError:
                        raw_args = {}
                raw_args = raw_args or {}
                signature = (name, json.dumps(raw_args, sort_keys=True, default=str))
                if signature in seen:
                    continue
                seen.add(signature)
                fresh.append((name, raw_args))

            if not fresh:
                messages.append({"role": "assistant", "content": text or ""})
                messages.append({"role": "user", "content": (
                    "Stop calling tools. Write the final answer from the tool results "
                    "above, quoting their numbers exactly.")})
                return self._text(self._chat_safe(messages, model, tools=False))

            messages.append({"role": "assistant", "content": text, "tool_calls": calls})
            for name, raw_args in fresh:
                tool_result = self.call_tool(name, raw_args)
                result["tool_calls"].append({"name": name, "arguments": raw_args})
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(tool_result, default=str)})
        messages.append({"role": "user", "content": (
            "Stop calling tools. Write the final answer from the tool results above.")})
        return prose or self._text(self._chat_safe(messages, model, tools=False))

    # -- public API --------------------------------------------------------
    def answer(self, question: str, context: Optional[Dict[str, Any]] = None,
               use_llm: bool = True, use_tools: bool = False) -> Dict[str, Any]:
        """Answer a question.  Facts always come from the trained model."""
        facts = self.engine.answer(question, context)
        status = self.status()
        result: Dict[str, Any] = {
            "question": question,
            "answer": facts["answer"],
            "deterministic_answer": facts["answer"],
            "evidence": facts["evidence"],
            "details": facts.get("details", {}),
            "source": "local-deterministic",
            "mode": "tools" if use_tools else "explain",
            "ollama": {"available": status["available"], "model": status["model"],
                       "error": status["error"]},
            "tool_calls": [],
            "audit": [],
        }
        if not use_llm or not status["available"] or not status["model"]:
            if not status["available"]:
                result["note"] = ("Ollama is not running, so this answer is the deterministic "
                                  "engine output (computed from the data and the trained "
                                  "model). Start Ollama for a written explanation.")
            elif not status["model"]:
                result["note"] = status["error"]
            return result

        started = time.time()
        try:
            if use_tools:
                prose = self._tool_loop(status["model"], question, context, facts, result)
            else:
                prose = self._explain(status["model"], question, facts)
            result["elapsed_seconds"] = round(time.time() - started, 2)

            if not prose:
                result["note"] = ("The language model returned no text, so the deterministic "
                                  "answer computed from the trained model is shown.")
                return result

            audit = self._numeric_audit(prose, facts)
            wording = self._wording_audit(prose, facts)
            result["audit"] = audit + wording

            if (audit or wording) and not use_tools:
                # One corrective retry: tell the model exactly what was rejected and why.
                retry = self._explain(status["model"], question, facts, prose,
                                      audit + wording)
                retry_audit = self._numeric_audit(retry, facts)
                retry_wording = self._wording_audit(retry, facts)
                if retry and not retry_audit and not retry_wording:
                    prose = retry
                    audit, wording = [], []
                    result["audit"] = []
                    result["retried"] = True
                else:
                    result["rejected_prose"] = prose
                    result["note"] = ("The language model went beyond the model output "
                                      "(numbers or causal wording), so the deterministic "
                                      "answer from the trained model is shown instead.")
                    return result

            if audit or wording:
                result["rejected_prose"] = prose
                result["note"] = ("The language model went beyond the model output (numbers or "
                                  "causal wording), so the deterministic answer is shown "
                                  "instead.")
                return result

            result["answer"] = prose
            result["source"] = "ollama+tools" if use_tools else "ollama+explanation"
            result["audit"] = []
        except OllamaUnavailable as exc:
            self.last_error = str(exc)
            result["note"] = (f"Ollama call failed ({exc}). Showing the deterministic answer "
                              f"computed from the trained model.")
            result["source"] = "local-deterministic"
        return result

    # -- prompts -----------------------------------------------------------
    @staticmethod
    def _system_prompt(facts: Dict[str, Any]) -> str:
        grounding = json.dumps({
            "deterministic_answer": facts["answer"],
            "evidence": facts["evidence"],
            "details": facts.get("details", {}),
        }, default=str)[:6000]
        return (f"{SYSTEM_PROMPT}\n\nAPPLICATION-GENERATED FACTS for this question (the only "
                f"permitted source of numbers; tools return the same values):\n{grounding}")

    @staticmethod
    def _user_prompt(question: str, context: Optional[Dict[str, Any]]) -> str:
        if context:
            return (f"{question}\n\nStudent context supplied by the interface: "
                    f"{json.dumps(context, default=str)}")
        return question

    @staticmethod
    def _wording_audit(text: str, facts: Dict[str, Any]) -> List[str]:
        """Flag causal phrasing the measured data does not support."""
        reference = json.dumps(facts, default=str).lower()
        flagged = []
        for phrase in ("impact", "causes", "causal", "caused by", "leads to", "boosts marks",
                       "raises marks", "improves marks"):
            if phrase in text.lower() and phrase not in reference:
                flagged.append(f"unsupported phrasing: {phrase!r}")
        return flagged

    @staticmethod
    def _numeric_audit(text: str, facts: Dict[str, Any]) -> List[str]:
        """List score-like numbers in the LLM text that are absent from the facts.

        Numbers that are meaningful on their own (score, grade probability, metric) must
        exist in the facts.  Trivial tokens (0-9) are ignored: they are list positions,
        grade letters' indices, or the "0-100" range and are never model claims.
        """
        import re

        allowed = set()
        for token in re.findall(r"-?\d+(?:\.\d+)?", json.dumps(facts, default=str)):
            try:
                allowed.add(round(float(token), 1))
            except ValueError:
                continue
        flags = []
        for token in re.findall(r"\b\d{1,3}(?:\.\d+)?\b", text):
            try:
                value = round(float(token), 1)
            except ValueError:
                continue
            if value < 10 and value == int(value):
                continue  # trivially small integers: list indices, "0-100", grade order
            if value not in allowed:
                flags.append(token)
        return sorted(set(flags))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ollama bridge for the student model.")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--ask")
    parser.add_argument("--context")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--tools", action="store_true",
                        help="let the language model call tools itself (opt-in)")
    args = parser.parse_args()
    service = OllamaService()
    if args.status:
        print(json.dumps(service.status(refresh=True), indent=2))
        return 0
    if args.ask:
        context = json.loads(args.context) if args.context else None
        result = service.answer(args.ask, context, use_llm=not args.no_llm,
                                use_tools=args.tools)
        print(json.dumps(result, indent=2, default=str))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
