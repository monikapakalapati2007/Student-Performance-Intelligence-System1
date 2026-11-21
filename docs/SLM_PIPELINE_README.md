# Student Performance SLM — trained model, full pipeline, Ollama demo

A compact decoder-only causal Transformer (RMSNorm + RoPE + SwiGLU, weight-tied head)
**trained from scratch** on student-performance records — plus the complete reproducible
pipeline that trains, evaluates, exports and serves it.

Everything is real and measured on this machine: no pretrained weights, no downloaded
tokenizer, no public text corpus, no fabricated metrics.

---

## ⚠️ Read this first: the data is student performance, not air quality

The project this pipeline was originally specified for was an **Air Quality SLM**, and the
brief explicitly said *never use Student Performance datasets*. The data actually supplied
is the opposite, so the pipeline now supports both domains via a **dataset profile** and was
trained on the data that exists:

| dataset | rows available | used | profile |
|---|---|---|---|
| `student_performance 1 million` (Kaggle: `kayhanh`) — `hugging face/million_*.parquet` | 1,000,000 | **train 20,000 / val 2,000 / test 2,000** | `student_performance` |
| `Students Performance in Exams` (Kaggle: `spscientist`) — `hugging face/exams_*.parquet` | 1,000 | not used (smaller table family) | `student_performance` |

The two families have different schemas, so the loader keeps the largest consistent family
and reports the choice. The word **Student Performance** appears in every report, the paper
title, the model card and the demo — nothing claims to be an air quality model.

The air-quality path is still present and fully working (`--profile air_quality`); it simply
has no data to train on. Add an air quality table and it runs unchanged.

---

## Results at a glance (measured, not estimated)

**Model** — 2,557,632 parameters, 4 layers, 4 heads, d_model 192, context 128,
vocabulary 4096 (byte-level BPE learned from the corpus), dropout 0.1.

**Training** — CPU only (no CUDA, no Apple MPS), `cpu_tiny` preset, batch 16, 900 steps
(one partial epoch of 119,463 blocks), AdamW + warm-up/cosine, 23 minutes wall clock.

**Test split** (2,000 held-out students, 1,517,269 tokens scored, never seen in training):

| metric | value |
|---|---|
| cross-entropy loss | **0.1786** |
| perplexity | **1.196** |
| teacher-forced token accuracy | **94.44 %** |
| QA mean token F1 (12 held-out questions) | 0.8667 |
| QA exact match | 0.1667 |
| QA numeric recall | **0.0** |
| numerals generated that the reference does not contain | 6 |

The last two rows are the honest headline. The model reproduces the *shape* of an answer
almost perfectly — template, field names, units — but assigns the wrong digits. It has
learned the corpus distribution, not record retrieval. That gap is reported everywhere
rather than hidden behind token accuracy.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

python src/train_slm.py                # train + evaluate + reports
python src/make_paper.py               # IEEE-format article
python src/export_ollama.py --verify   # GGUF + Modelfile
python src/chat_demo.py                # browser chat at http://127.0.0.1:8080
```

> **This machine:** the `python` on `PATH` is Python **2.7** (MGLTools) and cannot run this
> project. Use `.venv\Scripts\python.exe` or activate the venv first.

---

## Full lifecycle: data → model → Ollama chat demo

### Step 1 — Data collection and validation

Place tables in `data/processed/`, `data/training/`, or anywhere the loader searches.
Originals are opened **read-only**.

The loader walks the directories, reads every parquet/csv/tsv/json/jsonl/xlsx/feather file,
detects which profile each table belongs to, keeps the largest single-schema family, and
audits shape, dtypes, missing values, duplicates, distributions, 3×IQR outliers and
**cross-split record overlap**. Result: `reports/dataset_analysis.md`.

### Step 2 — Preprocessing and text serialisation

Cleaning is stateless and per-value; no statistic is fitted on one split and applied to
another. Exact duplicates are removed within a split only. Missing values are **omitted,
not imputed**. Outliers are **reported, not removed**. Each row becomes a record card plus
deterministic Q/A pairs quoting that row's own values:

```text
### Student record: for student 450579. Grade: A. Measurements: Weekly self study hours
18.6 hours per week; Attendance percentage 85 %; Class participation 7.1 points out of 10;
Total score 100 points out of 100. ### End
```

Frozen corpus: `data/processed/student_performance/{train,validation,test}.jsonl`.

### Step 3 — Tokenizer

Byte-level BPE learned from the training split only (`--vocab-size`, default 4096). Byte-level
means no out-of-vocabulary input. Saved to `models/student_performance_slm/tokenizer/`.

### Step 4 — Model and sanity checks

Hardware is detected and the preset, batch size, context and step budget follow from it.
Nine checks must pass before training starts: dataset non-empty, tokenizer round-trip,
tensor shapes and one-token shift, forward pass, finite loss, **causal mask** (prefix logits
unchanged when future tokens are appended — evaluated with dropout disabled so it measures
masking and not noise), finite gradients, weight update, loss decrease, exact checkpoint
round-trip.

### Step 5 — Training

```bash
python src/train_slm.py                       # full run
python src/train_slm.py --resume              # continue from checkpoints/latest_checkpoint.pt
python src/train_slm.py --reports-only        # rewrite markdown reports from saved metadata
```

AdamW, warm-up + cosine, gradient clipping, validation every 50 steps, best-on-validation
checkpoint, early stopping, resumable checkpoint, seeded row subsampling for CPU budgets.

Outputs: `models/student_performance_slm/{best_model.pt, final_model.pt, config.json,
training_config.json, profile.json, tokenizer/}` and `checkpoints/latest_checkpoint.pt`.

### Step 6 — Evaluation

```bash
python src/evaluate_slm.py
```

Test loss, perplexity, teacher-forced token accuracy, plus generation scoring: token F1,
exact match, numeric recall, and unsupported-numeral count. →
`reports/evaluation_results.json`, `reports/sample_predictions.json`.

### Step 7 — Research article

```bash
python src/make_paper.py      # -> reports/research_paper.md
```

IEEE structure: abstract, index terms, eight numbered sections, eleven data tables
(dataset shape, parameter budget, hardware, descriptive statistics with SD/percentiles/
outliers, Pearson correlation matrix, per-epoch loss, per-checkpoint loss, test metrics,
qualitative examples) and twelve numbered references. Generated only from run artifacts, so
the numbers are always real. Paste into `IEEEtran` for two-column layout.

### Step 8 — Export to GGUF and import into Ollama

```bash
python src/export_ollama.py --verify
```

The model is bias-free RMSNorm + RoPE + SwiGLU — llama.cpp's `llama` architecture — so the
exporter rewrites the checkpoint into llama.cpp tensor names and writes a **GGUF v3 file by
hand**, with the tokenizer in GPT-2 BPE form (byte→unicode table + merge list). Verified
output: **6.50 MB, 39 tensors, `general.architecture = llama`**.

```bash
winget install Ollama.Ollama        # one time
cd models/ollama
ollama create student-performance-slm -f Modelfile
ollama run student-performance-slm
```

### Step 9 — Chat demo without Ollama

```bash
python src/chat_demo.py             # http://127.0.0.1:8080
```

Browser chat UI plus an **Ollama-compatible REST API** (`/api/chat`, `/api/generate` with
NDJSON streaming, `/api/tags`, `/api/ps`, `/health`), stdlib-only server. Same interface as
`ollama serve`, so the demo works even if the GGUF route is blocked.

### Step 10 — Publish to Hugging Face

```bash
pip install huggingface_hub && huggingface-cli login
python src/upload_huggingface.py --dry-run
python src/upload_huggingface.py \
    --repo-id <username>/student-performance-slm \
    --dataset-repo-id <username>/student-performance-processed
```

Generates model and dataset cards from measured metrics. Nothing uploads without an explicit
`--repo-id`; no token is ever stored in the repo. Note the weights are a **custom
architecture** — `AutoModel.from_pretrained()` will not load them; the card includes the
loading snippet.

### Step 11 — GitHub

```bash
git init && git add . && git commit -m "Student Performance SLM pipeline"
git branch -M main
git remote add origin https://github.com/<username>/student-performance-slm.git
git push -u origin main
```

`.gitignore` excludes `.venv/`, weights, processed data, logs and the unrelated datasets, so
only source and docs are committed.

---

## Project structure

```
src/
    train_slm.py              main entry point: the whole pipeline
    profiles.py               dataset profiles (air_quality | student_performance)
    preprocess_air_quality.py cleaning, serialisation, QA generation, domain text
    dataset_loader.py         discovery, profile detection, validation, statistics
    tokenizer.py              byte-level BPE from scratch + block creation
    model.py                  config, hardware presets, checkpoint save/load
    transformer.py            RMSNorm, RoPE, causal attention, SwiGLU, GPT
    evaluate_slm.py           test metrics + QA probes
    inference.py              prompted generation, scoring, REPL
    export_ollama.py          GGUF v3 writer + Modelfile generator
    chat_demo.py              browser chat + Ollama-compatible REST API
    upload_huggingface.py     model/dataset cards + upload
    make_paper.py             IEEE-format article generator
    reporting.py              training_summary.md / dataset_analysis.md
    utils.py                  seeding, hardware detection, logging
docs/
    PRESENTATION_CHECKLIST.md the five deliverables, mapped to commands
```

Generated by the run (not committed): `models/`, `checkpoints/`, `reports/`,
`data/processed/`.

---

## Command reference

| command | purpose |
|---|---|
| `python src/train_slm.py` | validate → preprocess → tokenize → sanity → train → evaluate → reports |
| `--profile {auto,air_quality,student_performance}` | dataset domain (default `auto`) |
| `--resume` / `--reports-only` | resume training / regenerate markdown only |
| `--max-records N` `--max-eval-records N` | seeded subsample caps (`0` = all rows) |
| `--preset {cpu_tiny,cpu_small,mps_small,gpu_small}` | model size |
| `python src/evaluate_slm.py` | metrics + QA probes from saved artifacts |
| `python src/inference.py --question "..."` / `--interactive` | generation |
| `python src/chat_demo.py` | chat UI + Ollama-compatible API |
| `python src/export_ollama.py --verify` | GGUF + Modelfile |
| `python src/make_paper.py` | IEEE-format article |
| `python src/upload_huggingface.py --dry-run` | preview HF upload |

---

## Design notes

**Why a custom tokenizer.** CPU-only, offline, no downloaded artefacts; byte-level BPE keeps
the vocabulary small and never yields an unknown token.

**Why RoPE / RMSNorm / SwiGLU.** Modern standard, and together they *are* the `llama`
architecture — which is what makes the Ollama export possible without new kernels.

**Why numeric recall is reported.** A model this small learns answer shape far more easily
than digits. Token accuracy would hide that; numeric recall (measured: 0.0) exposes it.

**Why subsampling is logged.** 1,000,000 rows cannot be trained on in a CPU demo, so a seeded
subsample is used and recorded in `run_metadata.json` and both reports. Splits are never
re-partitioned, so validation and test stay disjoint.

---

## Limitations

- **Numbers are unreliable.** Numeric recall on held-out QA is 0.0; the model reproduces
  memorised values rather than reading the prompt. Do not use generated figures for decisions.
- **No retrieval.** Record-specific questions would require the record in the prompt.
- **Template bias.** Q/A pairs are generated deterministically, so evaluation measures
  fidelity to a templated distribution, not to human-written questions.
- **Demo scale.** 2.56M parameters, 900 steps, 20k records: a demonstration model, not a
  student-analytics system.
- **CPU only.** GPU/MPS presets exist but were not measured here.
- **Ollama load path untested.** The GGUF parses correctly, but `ollama create` could not be
  executed because Ollama is not installed on this machine.
