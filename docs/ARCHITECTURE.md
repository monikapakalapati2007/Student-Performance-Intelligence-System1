# Architecture

## What this project actually is

Two different kinds of model live in this repository. They must not be confused:

| Component | Kind | Framework / format | Role |
|---|---|---|---|
| `models/student_performance/best_model/model.joblib` | `HistGradientBoostingRegressor` (scikit-learn) | joblib | **primary**: predicts `total_score` |
| `.../grade_model.joblib` | `HistGradientBoostingClassifier` | joblib | predicts the letter grade from behaviour only |
| `.../exams_model.joblib` | `LinearRegression` on one-hot exam features | joblib | secondary maths-score reference model |
| `models/student_performance_slm/final_model.pt` | 4-layer decoder-only Transformer trained from scratch (PyTorch, 2.56 M params) | `.pt` | **experiment**, preserved, not used for prediction |
| `models/ollama/student-performance-slm.f16.gguf` | the same experimental Transformer exported to GGUF | GGUF (llama arch) | legacy artefact, not a usable adviser |
| Ollama model `student-performance-advisor` | llama-family LLM run by Ollama | Ollama | **language layer**: explains model output |

The primary predictor is **not** a language model and is not converted to GGUF. GGUF conversion
is only meaningful (and only performed) for the PyTorch Transformer experiment, which is a
genuine decoder-only architecture.

## Runtime architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ Browser  (app/static/index.html + app.js + styles.css)               │
└───────────────┬──────────────────────────────────────────────────────┘
                │ fetch  (JSON over HTTP, 127.0.0.1)
┌───────────────▼──────────────────────────────────────────────────────┐
│ app/server.py — standard-library ThreadingHTTPServer                 │
│   GET  /            static UI                                        │
│   GET  /api/health   service + Ollama status                         │
│   GET  /api/meta     measured metrics, importance, dataset stats     │
│   POST /api/predict  {study_hours, attendance, participation}        │
│   POST /api/ask      {question, context}                             │
│   POST /api/exams    five categorical exam features                  │
└───────┬───────────────────────────────────┬──────────────────────────┘
        │                                   │
┌───────▼──────────────┐          ┌─────────▼──────────────────────────┐
│ src/insight.py       │          │ src/ollama_service.py              │
│ InsightEngine        │◄─────────│ OllamaService (tool calling)       │
│ • validate inputs    │  tool    │ • /api/tags, /api/chat             │
│ • load joblib models │  results │ • executes 6 tools locally         │
│ • predict + what-if  │          │ • numeric audit of the LLM text    │
│ • dataset statistics │          │ • deterministic fallback           │
└───────┬──────────────┘          └─────────┬──────────────────────────┘
        │                                   │ HTTP
┌───────▼──────────────┐          ┌─────────▼──────────────────────────┐
│ models/student_      │          │ Ollama daemon 127.0.0.1:11434      │
│ performance/         │          │ model: student-performance-advisor │
│  best_model/*.joblib │          │ base : llama3.2:1b                 │
│  preprocessor/*.joblib│         └────────────────────────────────────┘
└──────────────────────┘
```

## Data flow for a prediction

1. `predict.py::Predictor` loads the saved preprocessor, regressor, grade classifier and the
   exams model. Nothing is fitted at inference time.
2. `InsightEngine.validate` enforces the documented input domains (hours 0–40, attendance
   0–100, participation 0–10) and rejects NaN/inf.
3. `Predictor.predict_frame` re-creates the three engineered features with the *training*
   formula (`src/preprocess.engineer_features`), keeps the training column order, and calls
   the regressor. The score is clipped to 0–100.
4. The grade classifier receives only the behaviour features (never the score), and returns
   per-grade probabilities.
5. `InsightEngine` adds the expected range (± test MAE), the wording band, permutation
   importance from `results/metrics.json`, and what-if estimates produced by re-running the
   regressor at raised study hours / attendance.

## Language layer contract

`OllamaService.answer()`:

1. Computes the deterministic answer first (`InsightEngine.answer`). This is the ground truth
   and is always shown, even when the LLM works.
2. If Ollama is reachable, sends the system prompt + facts + question, with six tool
   definitions. Tool calls are executed **locally** in `call_tool()`, which delegates to the
   same `InsightEngine`. The LLM never computes a score.
3. If the model returns prose, that prose becomes the answer and the deterministic answer is
   kept in `details` for the UI's evidence panel.
4. `_numeric_audit()` lists score-like numbers in the LLM prose that do not appear in the
   facts, so a drift is visible instead of hidden.
5. If Ollama is missing, slow or errors, the deterministic answer is returned with
   `source = "local-deterministic"` and an explicit note. The application never breaks.

## Module map

| File | Responsibility |
|---|---|
| `src/inspect_data.py` | dataset discovery, schema/quality report |
| `src/preprocess.py` | splits, cleaning, feature engineering, encoders, leakage guards |
| `src/train_model.py` | single training entry point (sanity check → candidates → search → refit → evaluate) |
| `src/evaluate.py` | metrics, plots, re-runnable evaluation |
| `src/predict.py` | inference API + CLI |
| `src/insight.py` | grounded predictions, statistics, improvements, deterministic answers |
| `src/ollama_service.py` | Ollama client, tools, agent loop, fallback, audit |
| `src/ollama_prompt.py` | canonical system prompt / template / parameters |
| `src/ollama_setup.py` | pull base model, render Modelfile, register model |
| `src/nl_demo.py` | earlier template NL demo (kept, independent) |
| `src/train_slm.py`, `model.py`, `tokenizer.py`, `transformer.py`, `export_ollama.py` | the original from-scratch SLM experiment |
| `app/server.py` | HTTP application |
| `tests/test_system.py` | end-to-end suite |

## Leakage guards

- `student_id` is never a feature.
- `total_score` is never an input to the grade classifier.
- The scaler and the one-hot encoder are fitted on the training split only.
- `grade` is never an input to the regressor.
- The supplied splits are used as-is after verifying zero `student_id` overlap.
