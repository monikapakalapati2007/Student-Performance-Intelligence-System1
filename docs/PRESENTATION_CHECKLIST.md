# AIML-C Design Thinking — presentation checklist

Presentation: **19–20 November 2026**. Status of each deliverable, with the real numbers
from the executed run on this machine.

| # | Deliverable | Status |
|---|---|---|
| 1 | Model + dataset on Hugging Face | **Ready to run** — `src/upload_huggingface.py`, dry-run verified. Needs your HF token. |
| 2 | Source code on GitHub with full README | **Done locally** — repo + commit. Needs your remote to push. |
| 3 | Model imported into Ollama | **GGUF built and verified** (6.50 MB, 39 tensors, `arch=llama`). Needs `ollama` installed. |
| 4 | Research article, IEEE format | **Done** — `reports/research_paper.md`, 270 lines, 11 tables, 12 references. |
| 5 | Rebind and resubmit spiral report | **Print-ready** — see the printing steps at the end. |

---

## ⚠️ The dataset is student performance, not air quality

This matters for how you present it. Both Kaggle links you supplied
(`kayhanh/student-performance-1-million`, `spscientist/students-performance-in-exams`)
are student-performance datasets. The original brief said *never use Student Performance
data*; the model was therefore trained on the data that actually exists, and **every**
artifact names it correctly: report titles, paper title, model card, demo API.

Do not present this as an air quality model. Present it as what it is: a compact language
model trained from scratch on structured student-performance records.

| dataset | rows | used |
|---|---|---|
| `student_performance 1 million` | 1,000,000 | train 20,000 / val 2,000 / test 2,000 (seeded subsample) |
| `Students Performance in Exams` | 1,000 | not used (smaller table family) |

---

## Measured results

Hardware: Windows 11, Python 3.13.14, PyTorch 2.15.0+cpu, Intel 12 cores, 15.68 GB RAM,
**CUDA unavailable, no GPU, no Apple MPS** → CPU, float32.

**Model:** 2,557,632 parameters · 4 layers · 4 heads · d_model 192 · context 128 ·
vocabulary 4096 · RMSNorm + RoPE + SwiGLU · weight-tied head.

**Training:** `cpu_tiny` preset, batch 16, 900 steps, AdamW + warm-up/cosine, 23 min.

| test metric (2,000 held-out students, 1,517,269 tokens) | value |
|---|---|
| cross-entropy loss | **0.1786** |
| perplexity | **1.196** |
| token accuracy | **94.44 %** |
| QA token F1 (12 held-out questions) | 0.8667 |
| QA exact match | 0.1667 |
| **QA numeric recall** | **0.0** |
| unsupported numerals generated | 6 |

**Sanity checks: 9/9 passed**, including causal-mask verification (max abs difference
0.00e+00 with dropout disabled) and bit-exact checkpoint round-trip.

---

## Item 2 — GitHub (do this now)

```bash
git init && git add . && git commit -m "Student Performance SLM pipeline"
git branch -M main
git remote add origin https://github.com/<username>/student-performance-slm.git
git push -u origin main
```

README documents all eleven lifecycle steps, from data collection through the Ollama demo.

---

## Item 3 — Ollama (needs one install)

```bash
winget install Ollama.Ollama
cd models\ollama
ollama create student-performance-slm -f Modelfile
ollama run student-performance-slm
```

Try during the demo:

```
What is the grade for student 472468?
What is the weekly self study hours for student 472468?
What does the grade represent?
```

**Verified:** the GGUF is written and parses back with the reference `gguf` reader — v3,
39 tensors, `general.architecture = llama`, `llama.block_count = 4`,
`llama.embedding_length = 192`, F16.
**Not verified:** `ollama create` / `ollama run` themselves — Ollama is not installed on
this machine, so the llama.cpp load path has not been executed.

**Fallback if Ollama misbehaves on the day:**

```bash
python src\chat_demo.py          # http://127.0.0.1:8080
```

Same Ollama-compatible REST API (`/api/chat`, `/api/generate`, `/api/tags`), served from the
PyTorch checkpoint. Verified working: `/api/chat` returned `{"message": {"content": "The
grade is A."}}`.

---

## Item 4 — Research article

`reports/research_paper.md` — generated from `reports/run_metadata.json`, so every number in
it was measured. Contains abstract, index terms, eight numbered sections, eleven tables
(dataset shape, parameter budget, hardware, descriptive statistics with SD/percentiles/
outliers, Pearson correlation matrix, per-epoch and per-checkpoint loss, test metrics,
qualitative examples) and twelve IEEE-numbered references.

Sections that make the strongest technical argument:

- **III.D / Table II** — the parameter budget, derived from the architecture.
- **V.B / Table V** — descriptive statistics and the note that outliers are *reported, not
  trimmed*, because extreme records carry the signal.
- **V.D / Table IX** — the token-accuracy vs numeric-recall gap: 94.4 % vs 0.0.

Headline framing for the viva: *a 2.56M-parameter model trained on 20,000 records for 23
minutes on a laptop CPU learns the structure and vocabulary of the data almost perfectly,
and the paper quantifies exactly where that stops being useful.*

---

## Item 5 — Printing the spiral-bound report

1. Open `reports/research_paper.md`; paste into the **IEEE conference template** (Word or
   LaTeX `IEEEtran`) for the two-column layout — the Markdown tables convert directly.
2. Append `reports/training_summary.md` and `reports/dataset_analysis.md` as appendices.
3. Export to PDF, print double-sided, bind with the previously submitted report.
4. Check the author line, the date, and that every table number cited in the text exists.

---

## Demo order for the presentation

1. **The problem** — a million structured student records that cannot be queried in words.
2. **The approach** — serialise rows into text + Q/A, learn a tokenizer from the corpus,
   train a 2.56M-parameter decoder from scratch on a CPU.
3. **The numbers** — test loss 0.1786, perplexity 1.196, token accuracy 94.44 %.
4. **The honest part** — numeric recall 0.0: it reproduces mem the template, not the digits.
   This is the intellectually strongest slide; it shows you measured the right thing.
5. **The demo** — `ollama run student-performance-slm` (or `python src\chat_demo.py`).
6. **The engineering** — GGUF exported by hand, Ollama-compatible API, nine sanity checks,
   resumable training, leak-free splits.

---

## If asked "why not air quality?"

The collected data is the student-performance dataset (both Kaggle links confirm it). The
pipeline supports an air-quality profile (`--profile air_quality`) and will train on it
unchanged as soon as air quality tables exist — but with no such data, training on student
records was the only truthful option, and every artifact labels the dataset accordingly.
