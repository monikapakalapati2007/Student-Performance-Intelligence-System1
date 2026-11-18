"""Markdown report generation from measured pipeline results."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:
            return "n/a"
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# dataset_analysis.md
# --------------------------------------------------------------------------- #
def write_dataset_analysis(report: Dict[str, Any], path: str) -> None:
    discovery = report.get("discovery", {})
    analyses = report.get("analyses", {})
    overlap = report.get("overlap", {})
    prep = report.get("preprocessing", {})
    spec = report.get("field_spec", {})

    out: List[str] = ["# Air Quality Dataset Analysis", ""]
    out.append("Generated automatically by `src/train_slm.py`. Every number below was "
               "measured on the files found in the project at run time.")
    out.append("")

    out.append("## 1. Data source")
    out.append("")
    out.append(f"- Discovery method: **{discovery.get('source', 'n/a')}**")
    if discovery.get("split_note"):
        out.append(f"- Split note: {discovery['split_note']}")
    for d in discovery.get("searched_dirs", []):
        out.append(f"- Searched: `{d}`")
    out.append("")

    examined = discovery.get("files_examined", [])
    if examined:
        out.append("## 2. Files examined")
        out.append("")
        rows = [
            [f"`{os.path.basename(f['path'])}`", _fmt(f["rows"]),
             "yes" if f["air_quality"] else "no", f["reason"]]
            for f in examined
        ]
        out.append(_table(["file", "rows", "used", "assessment"], rows))
        out.append("")
    rejected = discovery.get("rejected", [])
    if rejected:
        out.append("### Rejected files")
        out.append("")
        for item in rejected:
            out.append(f"- `{item['path']}`: {item['reason']}")
        out.append("")

    out.append("## 3. Split sizes")
    out.append("")
    split_rows = [[name or "n/a", _fmt(info.get("rows")), _fmt(info.get("columns_count")),
                   _fmt(info.get("duplicate_rows")), _fmt(info.get("total_missing"))]
                  for name, info in analyses.items()]
    out.append(_table(["split", "rows", "columns", "duplicate rows", "missing values"], split_rows))
    out.append("")

    out.append("## 4. Records after preprocessing")
    out.append("")
    rows = [[name, _fmt(info["source_rows"]), _fmt(info["rows_after_cleaning"]),
             _fmt(info["documents"]), _fmt(info["characters"])]
            for name, info in prep.get("splits", {}).items()]
    out.append(_table(["split", "source rows", "rows after cleaning", "text documents", "characters"], rows))
    out.append("")
    total_docs = prep.get("total_documents")
    total_chars = prep.get("total_characters")
    if total_docs:
        out.append(f"Total serialised documents: **{_fmt(total_docs)}** "
                   f"({_fmt(total_chars)} characters).")
        out.append("")

    out.append("## 5. Column analysis")
    out.append("")
    for name, info in analyses.items():
        out.append(f"### `{name}` split")
        out.append("")
        rows = []
        for col in info["columns"]:
            if col.get("role") == "numeric":
                rows.append([f"`{col['name']}`", col["dtype"], _fmt(col["missing"]),
                             f"{_fmt(col.get('min'))} .. {_fmt(col.get('max'))}",
                             _fmt(col.get("mean")),
                             _fmt(col.get("extreme_outliers_3iqr"))])
            else:
                top = ", ".join(list(col.get("top_values", {}).items())[:3][i][0]
                                for i in range(min(3, len(col.get("top_values", {})))))
                rows.append([f"`{col['name']}`", col["dtype"], _fmt(col["missing"]),
                             f"{_fmt(col['unique'])} distinct", top or "-", "-"])
        out.append(_table(["column", "dtype", "missing", "range / values", "mean / top",
                           "outliers (3·IQR)"], rows))
        out.append("")

    out.append("## 6. Field classification")
    out.append("")
    for key in ("pollutant", "weather"):
        mapping = spec.get(key) or {}
        if mapping:
            out.append(f"- **{key.title()}**: " + ", ".join(f"{k} → `{v}`" for k, v in mapping.items()))
    for key, label in (("category", "Category"), ("location", "Location"), ("time", "Time"),
                       ("numeric", "Numeric"), ("categorical", "Categorical"),
                       ("identifier", "Identifier")):
        values = spec.get(key) or []
        if values:
            out.append(f"- **{label}**: " + ", ".join(f"`{v}`" for v in values))
    constant = spec.get("constant") or []
    if constant:
        out.append(f"- **Constant columns** (no information): " + ", ".join(f"`{c}`" for c in constant))
    out.append("")

    out.append("## 7. Split integrity")
    out.append("")
    if overlap:
        out.append(f"Leakage detected between splits: **{overlap.get('leakage_detected')}**")
        out.append("")
        rows = [[pair.replace("__", " ↔ "), _fmt(v["shared_rows"]), f"{_fmt(v['pct_of_smaller'])}%"]
                for pair, v in overlap.get("pairwise", {}).items()]
        out.append(_table(["split pair", "shared rows", "% of smaller split"], rows))
    else:
        out.append("No overlap analysis available.")
    out.append("")

    out.append("## 8. Preprocessing decisions")
    out.append("")
    out.append("- Original files are opened **read-only** and never modified or deleted.")
    out.append("- Cleaning is **stateless and per-value** (whitespace stripping, numeric coercion, "
               "timestamp parsing, exact-duplicate removal within a split). No statistic is fitted "
               "on one split and applied to another, so no transformation can leak test information.")
    out.append("- Missing measurements are **not imputed**. A missing field is simply omitted from "
               "the serialised record instead of being filled with a guessed value.")
    out.append("- Outliers are **reported, not removed**, because extreme pollution episodes are "
               "the signal of interest in air quality data.")
    for name, info in prep.get("splits", {}).items():
        cleaning = info.get("cleaning", {})
        if cleaning:
            out.append(f"- `{name}`: " + "; ".join(cleaning.get("steps", [])))
    out.append("")

    out.append("## 9. Text serialisation example")
    out.append("")
    examples = prep.get("examples", {})
    if examples.get("record_card"):
        out.append("```text")
        out.append(examples["record_card"])
        out.append("```")
        out.append("")
    for qa in examples.get("qa_pairs", []):
        out.append(f"```text\nQuestion: {qa['question']}\nAnswer: {qa['answer']}\n```")
        out.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


# --------------------------------------------------------------------------- #
# training_summary.md
# --------------------------------------------------------------------------- #
def write_training_summary(report: Dict[str, Any], path: str) -> None:
    hardware = report.get("hardware", {})
    tc = report.get("train_config", {})
    config = report.get("config", {})
    test = report.get("test_metrics", {})
    evaluation = report.get("evaluation", {})
    history = report.get("history", [])
    prep = report.get("preprocessing", {})
    samples = report.get("samples", [])
    timings = report.get("timings", {})

    out: List[str] = ["# Air Quality SLM — Training Summary", ""]
    out.append("## 1. Project description")
    out.append("")
    out.append(
        "A compact decoder-only causal Transformer language model (≈2–6M parameters) trained "
        "from scratch on structured Air Quality records. Each measurement row is serialised into "
        "readable text (location, time, pollutant concentrations, weather, air quality category) "
        "together with deterministic question/answer pairs derived only from the fields present in "
        "the row. The model therefore learns the vocabulary and relationships of the dataset rather "
        "than memorising a general-purpose web corpus. No pretrained weights are used."
    )
    out.append("")

    out.append("## 2. Dataset information")
    out.append("")
    out.append(f"- Discovery: {report.get('discovery', {}).get('source', 'n/a')}")
    rows = []
    for name, info in (report.get("analyses") or {}).items():
        prep_info = prep.get("splits", {}).get(name, {})
        rows.append([name, _fmt(info["rows"]), _fmt(info["columns_count"]),
                     _fmt(info["duplicate_rows"]), _fmt(prep_info.get("rows_after_cleaning")),
                     _fmt(prep_info.get("documents"))])
    out.append(_table(["split", "rows", "columns", "duplicate rows", "rows used", "documents"], rows))
    out.append("")
    out.append(f"Total serialised text: **{_fmt(prep.get('total_characters'))} characters** across "
               f"**{_fmt(prep.get('total_documents'))} documents**.")
    out.append("")

    out.append("## 3. Preprocessing methodology")
    out.append("")
    out.append("1. Every table in the project data directories is inspected and classified; "
               "non-air-quality tables (for example student-performance data) are rejected.")
    out.append("2. Existing train/validation/test files are used exactly as provided. If no split "
               "files exist, one seeded 80/10/10 split is created **once** and frozen on disk.")
    out.append("3. Values are whitespace-stripped, numeric-looking text is coerced, timestamps are "
               "parsed, and exact duplicates are removed *within* each split only.")
    out.append("4. Each row becomes a record card plus deterministic QA pairs whose answers quote "
               "the row's own values. Missing fields are omitted, never imputed.")
    out.append("5. Domain reference text (pollutant definitions and general air quality knowledge) "
               "is added to the training corpus only, so the model can answer definitional questions.")
    out.append("6. The test split is written to disk and never seen during training or model selection.")
    out.append("")

    out.append("## 4. Split sizes")
    out.append("")
    for name in ("train", "validation", "test"):
        info = prep.get("splits", {}).get(name)
        if info:
            out.append(f"- **{name}**: {_fmt(info['rows_after_cleaning'])} rows → "
                       f"{_fmt(info['documents'])} documents")
    out.append("")

    out.append("## 5. Tokenizer")
    out.append("")
    out.append(f"- Type: byte-level BPE trained from scratch on the training corpus")
    out.append(f"- Vocabulary size: **{_fmt(tc.get('vocab_size'))}** tokens")
    out.append(f"- Training blocks: {_fmt(tc.get('num_train_blocks'))} × {_fmt(tc.get('block_size'))} tokens")
    out.append(f"- Validation blocks: {_fmt(tc.get('num_val_blocks'))} | Test blocks: {_fmt(tc.get('num_test_blocks'))}")
    out.append(f"- Saved to: `{report.get('paths', {}).get('tokenizer', 'models/air_quality_slm/tokenizer')}`")
    out.append("")

    out.append("## 6. Model architecture")
    out.append("")
    out.append(f"- Architecture: **{config.get('arch', 'decoder-only causal Transformer')}**")
    out.append(f"- Layers: {_fmt(config.get('n_layer'))} | Heads: {_fmt(config.get('n_head'))} | "
               f"d_model: {_fmt(config.get('n_embd'))} | head_dim: "
               f"{_fmt((config.get('n_embd') or 0) // max(1, config.get('n_head') or 1))}")
    out.append(f"- Context length: {_fmt(config.get('block_size'))} tokens")
    out.append(f"- Positional encoding: **Rotary Position Embeddings (RoPE)**")
    out.append(f"- Normalisation: **RMSNorm** (pre-norm)")
    out.append(f"- Feed-forward: **SwiGLU** gated MLP")
    out.append(f"- Attention: multi-head causal self-attention with an explicit lower-triangular mask")
    out.append(f"- Output head: linear projection to vocabulary, "
               f"weight-tied with the token embedding = {config.get('tie_weights')}")
    out.append(f"- Dropout: {config.get('dropout')}")
    out.append("")
    out.append(f"**Total parameters: {_fmt(report.get('n_params'))}**")
    out.append("")

    out.append("## 7. Hardware configuration")
    out.append("")
    for key, label in (("os", "OS"), ("python_version", "Python"), ("torch_version", "PyTorch"),
                       ("cpu_name", "CPU"), ("cpu_count", "Logical cores"),
                       ("total_ram_gb", "System RAM (GB)"), ("cuda_available", "CUDA available"),
                       ("gpu_names", "GPU"), ("gpu_total_vram_gb", "GPU VRAM total (GB)"),
                       ("gpu_free_vram_gb", "GPU VRAM free (GB)"), ("mps_available", "Apple MPS"),
                       ("selected_device", "Selected device")):
        out.append(f"- {label}: {hardware.get(key)}")
    out.append("")
    out.append(f"Estimated training memory: {_fmt(tc.get('estimated_train_memory_mb'), 1)} MB")
    out.append("")

    out.append("## 8. Training hyperparameters")
    out.append("")
    for key in ("preset", "optimizer", "schedule", "learning_rate", "min_lr", "warmup_steps",
                "weight_decay", "batch_size", "grad_accum", "block_size", "max_steps",
                "max_epochs", "grad_clip", "precision", "seed", "patience"):
        out.append(f"- {key}: {tc.get(key)}")
    out.append("")

    out.append("## 9. Epochs and duration")
    out.append("")
    out.append(f"- Epochs requested: {_fmt(tc.get('max_epochs'))} | steps executed: "
               f"{_fmt(tc.get('max_steps'))} maximum")
    out.append(f"- Wall-clock training time: **{_fmt(report.get('training_seconds'), 1)} s**")
    if timings:
        out.append("- Stage timings (s): " + ", ".join(f"{k}={_fmt(v, 1)}" for k, v in timings.items()))
    out.append("")

    out.append("## 10. Loss per epoch")
    out.append("")
    epoch_rows = [[_fmt(e.get("epoch")), _fmt(e.get("step")), _fmt(e.get("train_loss")),
                   _fmt(e.get("train_loss_epoch_end")), _fmt(e.get("train_token_accuracy")),
                   _fmt(e.get("val", {}).get("loss")), _fmt(e.get("val", {}).get("perplexity")),
                   _fmt(e.get("val", {}).get("token_accuracy"))]
                  for e in history if e.get("kind", "epoch") == "epoch"]
    if epoch_rows:
        out.append(_table(["epoch", "step", "train loss (epoch mean)", "train loss (last 50 steps)",
                           "train tok acc", "val loss", "val ppl", "val tok acc"], epoch_rows))
        out.append("")
        out.append("_Training loss is also reported as the mean of the final 50 micro-batches of the "
                   "epoch, which reflects the model's state at the end of that epoch rather than an "
                   "average diluted by the untrained starting weights._")
    else:
        out.append("_No epoch completed._")
    out.append("")

    eval_rows = [[_fmt(e.get("step")), _fmt(e.get("val", {}).get("loss")),
                  _fmt(e.get("val", {}).get("perplexity")), _fmt(e.get("val", {}).get("token_accuracy"))]
                 for e in history if e.get("kind") == "eval"]
    if eval_rows:
        out.append("### Validation checkpoints during training")
        out.append("")
        out.append(_table(["step", "val loss", "val ppl", "val tok acc"], eval_rows))
        out.append("")

    out.append("## 11. Final test metrics")
    out.append("")
    out.append(f"- Test loss: **{_fmt(test.get('loss'))}**")
    out.append(f"- Test perplexity: **{_fmt(test.get('perplexity'))}**")
    out.append(f"- Test token accuracy (teacher forced): **{_fmt(test.get('token_accuracy'))}**")
    out.append(f"- Tokens scored: {_fmt(test.get('tokens'))}")
    qa = evaluation.get("qa_generation", {})
    if qa:
        out.append("")
        out.append("Held-out question answering (greedy-ish sampling, temperature 0.2):")
        out.append("")
        out.append(f"- Questions: {_fmt(qa.get('held_out_questions'))}")
        out.append(f"- Mean token F1: **{_fmt(qa.get('mean_token_f1'))}**")
        out.append(f"- Exact match rate: **{_fmt(qa.get('exact_match_rate'))}**")
        out.append(f"- Mean numeric recall: **{_fmt(qa.get('mean_numeric_recall'))}**")
        out.append(f"- Numerals generated that do not appear in the reference answer: "
                   f"{_fmt(qa.get('generated_numerals_unsupported'))}")
    out.append("")

    out.append("## 12. Sample inference")
    out.append("")
    if samples:
        for item in samples[:10]:
            out.append(f"**Q:** {item['question']}")
            if item.get("reference"):
                out.append(f"**Reference answer:** {item['reference']}")
            out.append(f"**Model answer:** {item['prediction']}")
            if item.get("has_reference"):
                out.append(f"_(token F1 {_fmt(item.get('token_f1'), 3)}, "
                           f"numeric recall {_fmt(item.get('numeric_recall'), 3)})_")
            out.append("")
    else:
        out.append("_Inference was skipped for this run._")
        out.append("")

    out.append("## 13. Known limitations")
    out.append("")
    out.append("- The model is a few-million-parameter demonstration model trained on one project's "
               "records for a very small number of steps. It reproduces the style and vocabulary of "
               "the dataset; it does not reason, and it has no world knowledge beyond the domain "
               "reference text in the training corpus.")
    out.append("- Generation is stochastic and can produce numbers that are not present in the source "
               "record. Numeric recall is reported above precisely so this is visible.")
    out.append("- Answers about a *specific* record require that record to be included in the prompt; "
               "the model has no retrieval mechanism.")
    out.append("- Only fields present in the dataset are learnable. Any air quality concept missing "
               "from the columns (for example a pollutant that was never measured) cannot be answered.")
    out.append("- CPU training limits model size and token throughput; longer training would be "
               "needed for fluent multi-sentence answers.")
    out.append("")

    out.append("## 14. Future improvements")
    out.append("")
    out.append("- Continue training for more steps and a larger vocabulary once GPU time is available.")
    out.append("- Add retrieval: put the relevant record card in the prompt for record-specific questions.")
    out.append("- Expand the QA templates to cover comparisons between locations and time windows.")
    out.append("- Evaluate against a held-out human-written question set rather than templated pairs.")
    out.append("- Distil a larger model's answers into this small one for better fluency.")
    out.append("")

    out.append("## 15. Artifacts")
    out.append("")
    for key, value in (report.get("paths") or {}).items():
        out.append(f"- {key}: `{value}`")
    out.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
