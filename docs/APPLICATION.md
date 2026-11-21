# Running the application

## Start

```bash
run_app.bat
# or
.venv\Scripts\python.exe app\server.py --open
```

Then open <http://127.0.0.1:8000>. Options: `--port 8080`, `--host 127.0.0.1`.

No web framework is used — only the Python standard library (`http.server`), scikit-learn,
joblib and pandas. Ollama is optional: without it the assistant answers from the deterministic
engine and the header chip shows **ollama: offline**.

## Screens

| Area | Contents |
|---|---|
| Header | project title, active regression model, Ollama status |
| Student profile | study hours (0–40), attendance (0–100), participation (0–10); slider and numeric box stay in sync |
| Prediction | predicted score, grade badge, grade confidence, expected range, plain-language interpretation |
| Grade probabilities | per-grade probability bars from the classifier |
| Supporting factors | permutation importance with dominant/negligible labels |
| Model what-if | predicted score and delta for +2/+4/+8 study hours and +10 attendance |
| Ask the assistant | free-text questions; answers show evidence type, source (Ollama + tools or deterministic) and tool names |
| Dataset & model | MAE/RMSE/R², grade accuracy, ROC-AUC, exams R², importance bars, dataset summary cards |
| Limitations | measured caveats |
| Exams model | secondary maths-score model form |

## API

```bash
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/api/meta
curl -s -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": 15, "attendance": 85, "participation": 6.5}'
curl -s -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which features are most important?"}'
```

Errors are JSON: `400` for invalid input (range, type, missing field), `404` unknown route,
`500` unexpected failure. The server is single-process, threaded, and loads the models once at
startup. All responses are `Cache-Control: no-store`.

## Tests

```bash
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

25 tests cover artifacts, metric consistency, single/batch prediction, input validation,
what-if reasoning, grounded answers, the HTTP API (including 400/403/404 paths) and the Ollama
layer's tool execution, fallback and numeric audit. The LLM-specific test is skipped
automatically when no Ollama model is available.

## Files added for the interface

```
app/server.py                    HTTP server + API
app/static/index.html            single-page UI
app/static/styles.css            styling
app/static/app.js                UI logic (vanilla JS, no build step)
run_app.bat                      launcher
tests/test_system.py             end-to-end tests
```
