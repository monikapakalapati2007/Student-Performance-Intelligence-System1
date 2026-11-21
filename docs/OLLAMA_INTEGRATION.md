# Ollama integration

## What Ollama does — and does not do

Ollama is the **language layer only**. It never computes, estimates or recalls a score; the
application computes the answer with the trained scikit-learn models first, and the language
model only writes it out. Two modes are implemented:

**`explain` (default, used by the app).** The most reliable path:

```
USER QUESTION
     |
     v
src/insight.py::InsightEngine   (validates input, loads joblib models, computes statistics)
     |
     v
AUTHORITATIVE ANSWER + FACTS    (numbers, expected range, grade probabilities, evidence type)
     |
     v
OLLAMA llama-family model at localhost:11434  (rewrites the facts in plain language)
     |
     v
AUDIT: any number or causal word not present in the facts?  -> discard prose, keep facts
     |
     v
FINAL RESPONSE + authoritative model output shown next to the summary
```

**`tools` (opt-in, `use_tools: true` / `--tools`).** Genuine tool calling, kept for the
demonstration of the architecture:

```
QUESTION -> OLLAMA chooses a tool -> OllamaService.call_tool
         -> InsightEngine -> joblib models -> tool result JSON -> role="tool" message
         -> OLLAMA writes the final answer -> audit
```

The tool loop is capped at three rounds, identical repeated calls are dropped, and if the model
keeps calling tools it is asked for a final prose answer over the results it already has.
llama3.2 occasionally emits malformed tool-call grammar (HTTP 500 from llama.cpp); the client
retries once without tools, which stays grounded because the facts are already in the prompt.

### Tools exposed to the model

| tool | implementation |
|---|---|
| `predict_student_performance` | `InsightEngine.predict` → saved regressor + grade classifier |
| `get_feature_importance` | permutation importance from `results/metrics.json` |
| `get_dataset_statistics` | statistics of `data/processed/tabular/million_test.parquet` |
| `get_model_metrics` | measured metrics from `results/metrics.json` |
| `get_improvement_suggestions` | what-if re-runs of the regressor + measured limitations |
| `get_system_limitations` | measured limitations |

### Grounding safeguards

1. The deterministic answer is computed **before** the LLM is called and is always returned in
   `deterministic_answer` and `details`. The UI shows the authoritative model output under the
   language-model summary.
2. The system prompts (`src/ollama_prompt.py`) forbid inventing numbers and forbid causal claims
   about attendance/participation.
3. `_numeric_audit()` rejects prose containing a score-like number that is absent from the facts
   (trivial 0-9 tokens such as list indices and the "0-100" range are ignored).
4. `_wording_audit()` rejects causal phrasing ("impact", "causes", "leads to", ...) that the
   measured facts do not support. A rejection returns the deterministic answer and records the
   rejected text in `rejected_prose`.
5. Any Ollama failure (not installed, not running, model missing, timeout, HTTP 500) falls back to
   the deterministic answer with `source = "local-deterministic"` and a note. The application
   keeps working.

## Setup

```bash
# 1. install Ollama (Windows): winget install --id Ollama.Ollama -e
#    (or download from https://ollama.com/download) and start the desktop app

# 2. check it
.venv\Scripts\python.exe src\ollama_setup.py --status

# 3. pull the small base model (llama3.2:1b, ~1.3 GB) and register the project model
.venv\Scripts\python.exe src\ollama_setup.py --all
```

`--all` runs `--pull`, `--write-modelfile` and `--create`. It produces:

- `models/ollama/Modelfile.student-performance` (generated, committed for reference)
- Ollama model **`student-performance-advisor`** built `FROM llama3.2:1b`

Manual equivalent:

```bash
ollama pull llama3.2:1b
ollama create student-performance-advisor -f models/ollama/Modelfile.student-performance
ollama run student-performance-advisor
```

## Model choice

`llama3.2:3b` is the recommended adviser: it follows the rewrite instructions noticeably better
than 1b and still runs inside 4 GB of VRAM on this machine (RTX 3050 Laptop, CUDA).
`llama3.2:1b` is supported and faster but produced looser wording in testing, which the audits
then rejected in favour of the deterministic answer. Switch with
`OLLAMA_BASE_MODEL=llama3.2:3b python src/ollama_setup.py --all`.

Configuration: `temperature 0.15`, `top_p 0.9`, `num_ctx 4096`, `num_predict 400`,
`repeat_penalty 1.05`. The template is the Llama 3 header format matching the base model.

## Environment variables

| variable | default | purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | daemon address |
| `OLLAMA_MODEL` | auto-detected | force a specific model name |
| `OLLAMA_BASE_MODEL` | `llama3.2:1b` | base model pulled/used by `ollama_setup.py` |

## CLI

```bash
.venv\Scripts\python.exe src\ollama_service.py --status
.venv\Scripts\python.exe src\ollama_service.py --ask "What factors influence student performance?"
.venv\Scripts\python.exe src\ollama_service.py --ask "Predict the score for 18 hours, 92% attendance, participation 8"
```

## The original SLM experiment

`models/student_performance_slm/final_model.pt` is a genuine decoder-only Transformer trained
from scratch (4 blocks, `n_embd` 192, 4 heads, SwiGLU, RoPE, RMSNorm, tied weights, 2.56 M
trainable parameters, vocab 4096, context 128). `src/export_ollama.py` converted it to
`models/ollama/student-performance-slm.f16.gguf` (llama architecture, 39 tensors, verified
readable). That path is preserved.

It is **not** used as the project's assistant because a 2.5 M-parameter model trained on a few
hundred thousand tokens cannot answer free-form questions reliably, and the earlier evaluation
measured numeric recall of 0 for score questions. The legacy Modelfile in `models/ollama/`
still carries the air-quality system prompt from the original project and is kept only as a
historical artefact — do **not** register it for student questions. The adviser model defined in
`models/ollama/Modelfile.student-performance` supersedes it.

Note: `models/student_performance_slm/config.json` previously reported `block_size: 192` while
the checkpoint stores 128; the file has been corrected to 128 to match the checkpoint, the GGUF
metadata and `training_config.json`.
