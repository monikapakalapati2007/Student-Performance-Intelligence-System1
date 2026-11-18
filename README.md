# Air Quality SLM — complete pipeline, from raw records to an Ollama chat demo

A compact, **from-scratch decoder-only causal Transformer** (RMSNorm + RoPE + SwiGLU,
weight-tied output head) trained on this project's Air Quality records — plus the full
reproducible pipeline that produces, evaluates, exports and serves it.

No pretrained weights are downloaded and no public text corpus is trained on. The
tokenizer is learned from your data, the model is trained from random initialisation,
and every number reported anywhere in this repository comes from an executed run.

---

## ⚠️ Current status: the Air Quality dataset is still missing

As committed, this folder contains **no air quality data**. The only tables present are
Student Performance datasets, which this project must not use:

| file | rows | what it is |
|---|---|---|
| `hugging face/exams_train.parquet` | 700 | UCI Student Performance (gender, parental education, lunch, math/reading/writing scores) |
| `hugging face/exams_validation.parquet` | 150 | same |
| `hugging face/exams_test.parquet` | 150 | same |
| `hugging face/million_train_.parquet` | 700,000 | synthetic student study/attendance/score data |
| `hugging face/million_validation.parquet` | 150,000 | same |
| `hugging face/million_test.parquet` | 150,000 | same |

Running the pipeline therefore stops immediately (exit code `2`) and prints every file it
inspected and why it was rejected. It does not fall back to student data, does not
download a substitute dataset, and does not fabricate results.

**Everything downstream of training — the Hugging Face upload, the GGUF file, the Ollama
model, the demo, and the paper's results tables — needs this data.** The code for all of
it is finished and tested; see *Full lifecycle* below.

### Putting the data in place

```
data/processed/train.parquet + validation.parquet + test.parquet   ← preferred
data/training/*.csv                                                ← split by filename
a single .csv / .parquet / .jsonl air-quality table                ← auto-split 80/10/10, frozen
```

Columns are detected automatically: PM2.5, PM10, NO2, NO, NOx, SO2, CO, O3, NH3,
benzene, toluene, xylene, AQI, pmi; meteorological fields (temperature, humidity, wind
speed/direction, pressure, rainfall, dew point, visibility, solar radiation); location
fields (city, station, site, state, region, district, zone, latitude, longitude); time
fields; and category/status fields. Unrecognised columns are still serialised — nothing
is dropped. Minimum: two air-quality signals.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                  # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

python src/train_slm.py                 # train + evaluate + reports
python src/make_paper.py                # IEEE-format research article
python src/export_ollama.py --verify    # GGUF + Modelfile for Ollama
python src/chat_demo.py                 # browser chat at http://127.0.0.1:8080
```

> **Note for this machine:** the `python` on `PATH` is Python **2.7** (from MGLTools), and
> it cannot run this project. Use `.venv\Scripts\python.exe` explicitly, or activate the
> virtual environment first.

---

## Full lifecycle: Data Collection → Ollama chat demo

### Step 0 — Data collection

Collect the air quality measurements (station or city, timestamp, pollutant
concentrations, meteorological context, index/category) and place the processed tables in
`data/processed/` (or `data/training/`). Original files are never modified.

### Step 1 — Dataset validation and audit

```bash
python src/train_slm.py
```

Before any training, the pipeline walks the data directories, reads every
parquet/csv/tsv/json/jsonl/xlsx/feather table, classifies each column, rejects
non-air-quality tables, and reports shape, dtypes, missing values, duplicates,
distributions, 3×IQR outlier counts, and **cross-split record overlap** (leakage).
The audit is written to `reports/dataset_analysis.md`.

### Step 2 — Preprocessing and text serialisation

Automatic, inside the same command. Cleaning is stateless and per-value (no statistic
fitted on one split and applied to another); exact duplicates are removed within a split
only; missing measurements are **omitted, not imputed**; outliers are **reported, not
removed**. Each row becomes a record card plus deterministic question/answer pairs whose
answers quote that row's own values, e.g.

```text
### Air quality record: in Delhi on 2024-01-01 12:00. Pollutants: PM2.5 85 micrograms per
cubic metre; PM10 120 micrograms per cubic metre; ... Air quality index: AQI 175.
Weather: Temperature 30 degrees Celsius; Humidity 65 %. Air quality category: Moderate. ### End
```

The frozen corpus is written to `data/processed/aq_slm/{train,validation,test}.jsonl`.
The test split is never read again during training or model selection.

### Step 3 — Tokenizer training

A byte-level BPE tokenizer is learned from the training corpus only (`--vocab-size`,
default 4096). Byte-level means no input is ever out-of-vocabulary, and learning it from
your own data means no download. Saved to `models/air_quality_slm/tokenizer/`.

### Step 4 — Model initialisation and sanity checks

Hardware is detected first (OS, Python, PyTorch, CPU, RAM, CUDA, GPU name/VRAM, Apple
MPS) and the preset, batch size, sequence length and step budget are chosen from what is
actually available. Parameter count and estimated training memory are printed before
training starts.

Nine sanity checks must then pass or training refuses to start: dataset non-empty,
tokenizer round-trip, tensor shapes and one-token shift, forward pass, finite loss,
**causal mask** (prefix logits unchanged when future tokens are appended), finite
gradients, weights change after one step, loss decreases over a fixed batch, and
bit-exact checkpoint round-trip.

### Step 5 — Training

```bash
python src/train_slm.py                 # full run
python src/train_slm.py --resume        # continue from checkpoints/latest_checkpoint.pt
```

AdamW, linear warm-up + cosine decay, gradient clipping, optional gradient accumulation
and fp16 autocast on CUDA, validation every 50 steps, best-on-validation checkpointing,
early stopping, and a resumable `latest_checkpoint.pt`. Per-step loss, learning rate and
throughput are logged to `reports/train.log`.

Outputs:

```
models/air_quality_slm/best_model.pt         lowest validation loss
models/air_quality_slm/final_model.pt        last step of the run
models/air_quality_slm/config.json           architecture
models/air_quality_slm/training_config.json  optimiser, schedule, hardware
models/air_quality_slm/tokenizer/            vocab.json, merges.json, tokenizer_config.json
checkpoints/latest_checkpoint.pt             resume point
```

### Step 6 — Evaluation

```bash
python src/evaluate.py
```

Test loss, perplexity, teacher-forced token accuracy, plus generation-based scoring of
held-out QA pairs: token F1, exact match, **numeric recall**, and a count of numerals the
model invented that the reference answer does not contain. Written to
`reports/evaluation_results.json` and `reports/sample_predictions.json`.

### Step 7 — Research article

```bash
python src/make_paper.py        # -> reports/research_paper.md
```

Generates the IEEE-format article from the run artifacts: abstract, index terms, eight
numbered sections, eleven data tables (dataset shape, parameter budget, hardware,
descriptive statistics with SD/percentiles/outliers, Pearson correlation matrix, per-epoch
and per-checkpoint loss, test metrics, qualitative examples) and twelve numbered IEEE
references. It cannot be generated without a completed run, so the numbers are always real.
Paste into the `IEEEtran` template for the final two-column layout.

### Step 8 — Export to GGUF and import into Ollama

```bash
python src/export_ollama.py --verify
```

The model is bias-free with RMSNorm, RoPE and SwiGLU — exactly llama.cpp's `llama`
architecture — so the exporter rewrites the checkpoint into llama.cpp tensor names
(`token_embd`, `blk.N.attn_q/k/v`, `blk.N.ffn_gate/up/down`, `output_norm`, `output`) and
writes a **GGUF v3 file by hand**, with the tokenizer in GPT-2 BPE form (byte→unicode
mapping and merge list) so Ollama tokenizes text identically to training. `--verify` reads
the file back and prints the parsed metadata.

Install Ollama (one time), then:

```bash
winget install Ollama.Ollama          # Windows;  or download from https://ollama.com
ollama serve                          # usually already running after install

cd models/ollama
ollama create air-quality-slm -f Modelfile
ollama run air-quality-slm
```

The generated `Modelfile` pins the exact training prompt format, `temperature 0.2`,
`top_k 40` and the stop markers, so answers do not degrade:

```
FROM ./air-quality-slm.f16.gguf
TEMPLATE """### Question and answer
Question: {{ .Prompt }}
Answer:"""
PARAMETER stop "### End"
```

### Step 9 — Chat demo without Ollama

If Ollama is not available, the same model can be served straight from the PyTorch
checkpoint:

```bash
python src/chat_demo.py                # http://127.0.0.1:8080
```

This starts a browser chat UI and an **Ollama-compatible REST API** — `POST /api/chat`,
`POST /api/generate` (streaming and non-streaming), `GET /api/tags`, `GET /api/ps`,
`GET /health` — using only the Python standard library for the server. Any Ollama client
or the same frontend works against it unchanged, which is useful if the GGUF route is
blocked on the day.

### Step 10 — Publish to Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login                       # paste a write token

python src/upload_huggingface.py --dry-run  # list exactly what would be uploaded
python src/upload_huggingface.py \
    --repo-id <your-username>/air-quality-slm \
    --dataset-repo-id <your-username>/air-quality-dataset
```

Generates a model card and a dataset card from the real artifacts (architecture table,
measured metrics, limitations), then uploads the weights, config, tokenizer, GGUF and
processed dataset. Nothing is uploaded without an explicit `--repo-id`, and no token is
ever written into the repository.

One caveat: this is a **custom architecture**, not a `transformers` class, so
`AutoModel.from_pretrained()` will not load the weights. The generated model card
includes the exact loading snippet instead. Making it loadable through `transformers`
requires registering the modelling code under `auto_map` with `trust_remote_code` —
worth doing only if you actually need it.

### Step 11 — Maintain the code on GitHub

```bash
git init
git add .
git commit -m "Air Quality SLM: training pipeline, evaluation, GGUF export, Ollama demo"
git branch -M main
git remote add origin https://github.com/<your-username>/air-quality-slm.git
git push -u origin main
```

`.gitignore` already excludes `.venv/`, `__pycache__/`, `checkpoints/*.pt`,
`models/**/*.pt`, `data/processed/` and the smoke-test fixtures, so only source, docs and
configuration are committed. The Student Performance parquet files are excluded too —
they are not part of this project.

---

## Project structure

```
src/
    train_slm.py              # main entry point: the whole pipeline
    preprocess_air_quality.py # cleaning, text serialisation, QA generation, domain text
    dataset_loader.py         # discovery, validation, field classification, statistics
    tokenizer.py              # byte-level BPE trained from scratch + block creation
    model.py                  # config, hardware presets, checkpoint save/load
    transformer.py            # RMSNorm, RoPE, causal attention, SwiGLU, GPT
    evaluate.py               # test-set metrics + QA probes
    inference.py              # prompted generation, scoring, interactive REPL
    export_ollama.py          # GGUF writer (v3) + Ollama Modelfile generator
    chat_demo.py              # browser chat + Ollama-compatible REST API
    upload_huggingface.py     # model/dataset cards + Hugging Face upload
    make_paper.py             # IEEE-format research article generator
    reporting.py              # training_summary.md / dataset_analysis.md generation
    utils.py                  # seeding, hardware detection, logging, JSON helpers
docs/
    PRESENTATION_CHECKLIST.md # the five deliverables, mapped to commands
models/air_quality_slm/       # produced by training (see Step 5)
models/ollama/                # produced by export_ollama.py
checkpoints/                  # produced by training
reports/                      # produced by training / evaluation / make_paper
data/processed/aq_slm/        # produced by preprocessing
```

Artifact folders are created **by the run**; they are not committed with placeholder
contents.

---

## Command reference

| command | purpose |
|---|---|
| `python src/train_slm.py` | full pipeline: validate → preprocess → tokenize → sanity → train → evaluate → reports |
| `python src/train_slm.py --resume` | continue from `checkpoints/latest_checkpoint.pt` |
| `python src/evaluate.py` | test metrics + QA probes from saved artifacts |
| `python src/inference.py` | sample questions |
| `python src/inference.py --question "..."` | one question |
| `python src/inference.py --interactive` | REPL |
| `python src/chat_demo.py` | browser chat + Ollama-compatible API |
| `python src/export_ollama.py --verify` | GGUF + Modelfile |
| `python src/make_paper.py` | IEEE-format article |
| `python src/upload_huggingface.py --dry-run` | preview the HF upload |

Useful `train_slm.py` flags: `--preset {cpu_tiny,cpu_small,mps_small,gpu_small,smoke}`,
`--data-dir`, `--workdir`, `--epochs`, `--max-steps`, `--batch-size`, `--block-size`,
`--grad-accum`, `--lr`, `--vocab-size`, `--device`, `--resume`, `--patience`, `--seed`.

---

## Design notes

**Why a custom tokenizer.** The pipeline must run on a CPU-only laptop with no network,
so a downloaded vocabulary is not acceptable. Byte-level BPE learned from the corpus
keeps the vocabulary small and makes every byte representable.

**Why RoPE, RMSNorm and SwiGLU.** They are the modern standard and, importantly, they are
the `llama` architecture — which is what makes Step 8 possible. A model built from
LayerNorm and learned positional embeddings could not be handed to llama.cpp without new
kernels.

**Why ask for numeric recall.** A model this small learns the *shape* of an answer far
more easily than the digits inside it. Token accuracy would hide that; numeric recall and
the unsupported-numeral count expose it.

---

## Honest expectations

This is a demonstration-scale model: roughly 2–6M parameters on the CPU presets, trained
on one project's records. It learns the vocabulary, format and correlations of the
dataset very well, but generated numbers can be wrong, record-specific questions need the
record in the prompt (there is no retrieval), and fluent free-form prose needs more
training than a CPU budget allows. Those limits are stated in the reports, the model card
and the paper rather than averaged away.
