# AIML-C Design Thinking — presentation checklist

Presentation date: **19–20 November 2026**.

| # | Deliverable | Code status | Blocked on |
|---|---|---|---|
| 1 | Trained SLM model + dataset on Hugging Face | ✅ ready (`src/upload_huggingface.py`) | **air quality dataset** |
| 2 | Source code on GitHub with full README | ✅ ready | nothing (just `git push`) |
| 3 | Model imported into Ollama for a live demo | ✅ ready (`src/export_ollama.py`) | **air quality dataset** + `ollama` install |
| 4 | Research article in IEEE format | ✅ ready (`src/make_paper.py`) | **air quality dataset** (results tables) |
| 5 | Rebind and resubmit the spiral-bound report | ✅ paper is print-ready once generated | **air quality dataset** |

Everything that does not require the dataset is finished and tested. Items 1, 3, 4 and 5
all depend on one thing: **the air quality data is not in the project**.

---

## ⚠️ The one blocker

The folder contains only Student Performance tables
(`hugging face/exams_*.parquet`, `hugging face/million_*.parquet`). The project brief
explicitly forbids using them, so the pipeline refuses to train and says why.

Until an air quality table is added, there is no trained model to host, import into
Ollama, or report results for. Nothing has been faked to fill the gap.

**Fastest fix:** drop any air quality table in `data/processed/` (or `data/training/`, or
the project root) as `.csv` / `.parquet` / `.jsonl`. A single file is enough — the
pipeline splits it 80/10/10 once and freezes that split.

---

## Item 2 — GitHub (can be done right now)

```bash
git init
git add .
git commit -m "Air Quality SLM: training pipeline, evaluation, GGUF export, Ollama demo"
git branch -M main
git remote add origin https://github.com/<your-username>/air-quality-slm.git
git push -u origin main
```

`.gitignore` already excludes `.venv/`, `__pycache__/`, model weights, processed data and
the Student Performance parquet files, so the repository contains source, docs and
configuration only.

The README already documents the whole lifecycle from data collection to the Ollama chat
demo (Steps 0–11).

---

## The five commands, once the data is in place

```bash
# 1. train + evaluate + reports   (~minutes on CPU)
.venv\Scripts\python.exe src\train_slm.py

# 2. IEEE-format research article -> reports/research_paper.md
.venv\Scripts\python.exe src\make_paper.py

# 3. GGUF + Modelfile for Ollama
.venv\Scripts\python.exe src\export_ollama.py --verify

# 4. import into Ollama and run the demo
cd models\ollama
ollama create air-quality-slm -f Modelfile
ollama run air-quality-slm
#    (or, if Ollama is unavailable:  python src\chat_demo.py  -> http://127.0.0.1:8080)

# 5. publish to Hugging Face
.venv\Scripts\python.exe src\upload_huggingface.py --dry-run
.venv\Scripts\python.exe src\upload_huggingface.py ^
    --repo-id <username>/air-quality-slm ^
    --dataset-repo-id <username>/air-quality-dataset
```

---

## What was verified on this machine

Measured, not assumed:

- **Hardware:** Windows 11, Python 3.13.14, PyTorch 2.15.0+cpu, Intel (12 logical cores),
  15.68 GB RAM, **CUDA unavailable**, **no GPU**, **no Apple MPS** → device `cpu`, fp32.
- **Sanity checks:** 9/9 passed (including causal-mask verification and exact checkpoint
  round-trip).
- **Training:** ran to completion, checkpoints saved, `--resume` from
  `checkpoints/latest_checkpoint.pt` verified.
- **Evaluation / inference / paper / GGUF / chat server:** all executed successfully.
- **GGUF:** written by hand, then read back with the reference `gguf` parser — GGUF v3,
  21 tensors, `general.architecture = llama`, `token_embd.weight` shape `[64, 1024]`,
  F16 payload.
- **Chat API:** `/health`, `/api/tags`, `/api/chat` (non-streaming) and `/api/generate`
  (NDJSON streaming) all returned valid Ollama-shaped responses.

These runs used a **synthetic fixture** purely to exercise the code; its results are not
reported anywhere and its artifacts were deleted. No model is shipped in `models/`.

**Not verified:** `ollama create` / `ollama run` against the GGUF file — Ollama is not
installed on this machine, so the llama.cpp load path could not be executed here. The GGUF
container and metadata parse correctly; the first real test will be
`ollama create ... -f Modelfile`.

---

## Contingency for the presentation

If the dataset cannot be located in time:

1. **Run the demo from the chat server** (`python src/chat_demo.py`) instead of Ollama.
   It exposes the same Ollama-compatible API from the PyTorch checkpoint, so the demo
   story is unchanged and no GGUF build is needed.
2. **Present the pipeline as the deliverable**, with `reports/dataset_analysis.md` from a
   real run of whatever data *is* available — but do not present numbers that were not
   measured on air quality data.
3. **Or authorize a public air quality dataset** (for example an open CPCB/OpenAQ
   station export). The brief says not to substitute one, so that is a decision to make
   explicitly, and the reports would then need to state clearly which dataset was used.

---

## Item 5 — spiral binding (physical)

Once `python src/make_paper.py` has run, the article is at `reports/research_paper.md`
(generated with the real numbers from the run). To print:

1. Paste into the **IEEE conference template** (Word or LaTeX `IEEEtran`) for the
   two-column layout — the eleven Markdown tables convert directly.
2. Export to PDF, print double-sided, and bind it together with the previously submitted
   report.
3. Check the print date, the author line, and that every table number in the text matches
   the table above it.

`reports/training_summary.md` and `reports/dataset_analysis.md` are the supporting
appendix material for the same submission.
