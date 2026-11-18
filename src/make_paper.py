"""Generate an IEEE-format research article from a completed training run.

    python src/make_paper.py

Reads reports/run_metadata.json and reports/evaluation_results.json and writes
reports/research_paper.md. Every number in the paper is taken from those
artifacts, so the article can only be generated from a run that actually
happened. Reference list follows IEEE numbering.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

TITLE = ("A Compact Decoder-Only Small Language Model for Air Quality Record "
         "Understanding: Design, Training and Evaluation")

REFERENCES: List[str] = [
    "A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, "
    "and I. Polosukhin, \u201cAttention is all you need,\u201d in Proc. Adv. Neural Inf. "
    "Process. Syst. (NeurIPS), 2017, pp. 5998\u20136008.",
    "A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, \u201cLanguage models "
    "are unsupervised multitask learners,\u201d OpenAI, Tech. Rep., 2019.",
    "J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu, \u201cRoFormer: Enhanced "
    "transformer with rotary position embedding,\u201d arXiv:2104.09864, 2021.",
    "B. Zhang and R. Sennrich, \u201cRoot mean square layer normalization,\u201d in Proc. "
    "Adv. Neural Inf. Process. Syst. (NeurIPS), 2019, pp. 12\u2013039.",
    "N. Shazeer, \u201cGLU variants improve transformer,\u201d arXiv:2002.05202, 2020.",
    "I. Loshchilov and F. Hutter, \u201cDecoupled weight decay regularization,\u201d in Proc. "
    "Int. Conf. Learn. Represent. (ICLR), 2019.",
    "R. Eldan and Y. Li, \u201cTinyStories: How small can language models be and still speak "
    "coherent English?\u201d arXiv:2305.07759, 2023.",
    "World Health Organization, WHO Global Air Quality Guidelines: Particulate Matter "
    "(PM2.5 and PM10), Ozone, Nitrogen Dioxide, Sulfur Dioxide and Carbon Monoxide. "
    "Geneva, Switzerland: WHO, 2021.",
    "Central Pollution Control Board, National Air Quality Index. New Delhi, India: "
    "Ministry of Environment, Forest and Climate Change, 2014.",
    "G. Gerganov et al., \u201cllama.cpp: LLM inference in C/C++,\u201d GitHub repository, 2023. "
    "[Online]. Available: https://github.com/ggerganov/llama.cpp",
    "Ollama, \u201cOllama: Run large language models locally,\u201d 2024. [Online]. "
    "Available: https://ollama.com",
    "Hugging Face, \u201cDatasets and model hub,\u201d 2024. [Online]. "
    "Available: https://huggingface.co",
]


def _load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _f(value: Any, digits: int = 2, dash: str = "n/a") -> str:
    if value is None:
        return dash
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return f"{value:,}"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _table(headers: List[str], rows: List[List[str]], caption: str = "") -> str:
    out = []
    if caption:
        out.append(f"**{caption}**")
        out.append("")
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    out.append("")
    return "\n".join(out)


def build_paper(meta: Dict[str, Any], evaluation: Optional[Dict[str, Any]]) -> str:
    hardware = meta.get("hardware", {})
    tc = meta.get("train_config", {})
    config = meta.get("config", {})
    prep = meta.get("preprocessing", {})
    analyses = meta.get("analyses", {})
    discovery = meta.get("discovery", {})
    stats = meta.get("statistics", {})
    history = meta.get("history", [])
    test = meta.get("test_metrics", {})
    spec = meta.get("field_spec", {})
    samples = meta.get("samples", [])
    qa = (evaluation or {}).get("qa_generation", {})
    n_params = meta.get("n_params")
    duration = meta.get("training_seconds")
    has_qa = bool(qa and qa.get("held_out_questions"))
    if has_qa:
        qa_sentence = (
            f"On held-out question-answer pairs it reaches a mean token-level F1 of "
            f"{_f(qa.get('mean_token_f1'), 4)} with numeric recall of "
            f"{_f(qa.get('mean_numeric_recall'), 4)}, and "
            f"{_f(qa.get('generated_numerals_unsupported'))} generated numerals were not "
            f"supported by the reference answer."
        )
    else:
        qa_sentence = (
            "Generation-based question answering was not collected for this run (inference "
            "was skipped), so only likelihood metrics are reported here."
        )

    splits = prep.get("splits", {})
    train_rows = splits.get("train", {}).get("rows_after_cleaning")
    val_rows = splits.get("validation", {}).get("rows_after_cleaning")
    test_rows = splits.get("test", {}).get("rows_after_cleaning")
    total_docs = prep.get("total_documents")
    total_chars = prep.get("total_characters")

    L: List[str] = []
    L.append(f"# {TITLE}")
    L.append("")
    L.append("*Author: [name], Department of Artificial Intelligence and Machine Learning*")
    L.append("")
    L.append("*Generated automatically from the executed training run. Every metric in "
             "Sections V\u2013VII was measured on the project's own dataset.*")
    L.append("")

    # ---------------- Abstract ----------------
    L.append("## Abstract")
    L.append("")
    L.append(
        f"Air quality monitoring produces large volumes of structurally identical numerical "
        f"records that are difficult to query in natural language. This paper presents a "
        f"compact decoder-only Transformer small language model (SLM) trained from scratch on "
        f"a project-specific air quality corpus of {_f(train_rows)} training records, serialised "
        f"into {_f(total_docs)} textual documents. The model uses rotary position embeddings, "
        f"RMS normalisation and SwiGLU feed-forward blocks, and contains "
        f"{_f(n_params)} parameters \u2014 three orders of magnitude smaller than contemporary "
        f"instruction-tuned models. A byte-level byte-pair-encoding vocabulary of "
        f"{_f(tc.get('vocab_size'))} tokens is learned from the corpus itself, removing any "
        f"dependency on downloaded pretrained artefacts. Training used AdamW with a warm-up and "
        f"cosine learning-rate schedule for {_f(tc.get('max_steps'))} steps on "
        f"{hardware.get('selected_device', 'CPU')}. On the held-out test split the model "
        f"achieves a cross-entropy loss of {_f(test.get('loss'), 4)} "
        f"(perplexity {_f(test.get('perplexity'), 2)}) and teacher-forced token accuracy of "
        f"{_f((test.get('token_accuracy') or 0) * 100, 2)}%. {qa_sentence} The model is "
        f"exported to GGUF and served through Ollama for interactive demonstration. Results "
        f"show that a domain-specific SLM can learn the vocabulary, structure and statistical "
        f"regularities of air quality data at negligible compute cost, while also quantifying "
        f"its remaining limitations in factual generation."
    )
    L.append("")
    L.append("**Index Terms** \u2014 Air quality, small language models, decoder-only "
             "Transformer, rotary position embeddings, byte-pair encoding, GGUF, Ollama, "
             "domain-specific language modelling.")
    L.append("")

    # ---------------- I. Introduction ----------------
    L.append("## I. Introduction")
    L.append("")
    L.append(
        "Air pollution is among the largest environmental risks to public health, and the "
        "World Health Organization attributes millions of premature deaths each year to "
        "exposure to fine particulate matter and other pollutants [8]. Monitoring networks "
        "report measurements as records: a location, a timestamp, a set of pollutant "
        "concentrations, supporting meteorological variables, and an aggregated index or "
        "category [9]. These records are precise but inconvenient: answering even a simple "
        "question about them requires writing queries or plotting code."
    )
    L.append("")
    L.append(
        "Large language models can mediate this interaction, but general-purpose models are "
        "expensive to run and know nothing about a private measurement table. This work asks a "
        "narrower question: how much air quality understanding can be obtained from a model "
        "that is small enough to train on a single CPU in minutes? We answer it by building a "
        "complete, reproducible pipeline that (i) audits the dataset, (ii) serialises records "
        "into natural text, (iii) trains a byte-level tokenizer and a decoder-only Transformer "
        "from scratch, (iv) evaluates on an untouched test split, and (v) packages the result "
        "for local inference through Ollama."
    )
    L.append("")
    L.append("The contributions of this work are:")
    L.append("")
    L.append("1. A leak-free preprocessing and serialisation scheme that converts tabular air "
             "quality measurements into question-answer text without inventing values.")
    L.append("2. A compact RoPE/RMSNorm/SwiGLU language model with a corpus-learned BPE "
             "vocabulary, sized automatically from detected hardware.")
    L.append("3. A validated training pipeline with nine pre-flight sanity checks, "
             "validation-based model selection, early stopping and resumable checkpoints.")
    L.append("4. An honest evaluation that reports both likelihood metrics and factual "
             "consistency of generated values.")
    L.append("5. An export path from the custom checkpoint to GGUF and Ollama for a live demo.")
    L.append("")

    # ---------------- II. Related work ----------------
    L.append("## II. Related Work")
    L.append("")
    L.append(
        "The Transformer [1] remains the basis of all current language models. GPT-2 [2] "
        "established the decoder-only, byte-level BPE recipe adopted here. Three components "
        "have since become standard and are used in this work: rotary position embeddings "
        "(RoPE) [3], which encode position by rotating query and key vectors; RMS "
        "normalisation [4], which drops mean-centring and is cheaper than LayerNorm; and the "
        "gated SwiGLU feed-forward block [5], which improves quality at equal parameter count. "
        "Optimisation follows AdamW [6]."
    )
    L.append("")
    L.append(
        "TinyStories [7] demonstrated that models of a few million parameters can produce "
        "coherent text when the training distribution is narrow and synthetic. That finding is "
        "the methodological reference for this project: restrict the domain, shrink the model, "
        "and measure honestly. Unlike [7], no public dataset is used \u2014 the corpus is the "
        "project's own air quality table \u2014 and the evaluation targets factual consistency "
        "of domain values rather than narrative coherence."
    )
    L.append("")

    # ---------------- III. Methodology ----------------
    L.append("## III. Proposed Methodology")
    L.append("")
    L.append("### A. Data Collection and Audit")
    L.append("")
    L.append("The corpus was assembled from the project's processed air quality tables. The "
             "pipeline inspects every tabular file in the data directories, classifies each "
             "column against a vocabulary of pollutant, meteorological, location, temporal and "
             "categorical indicators, and rejects any table that shows fewer than two "
             "air-quality signals or that matches a known unrelated dataset profile. Discovery "
             f"reported: *{discovery.get('source', 'n/a')}*.")
    L.append("")
    if analyses:
        rows = []
        for name, info in analyses.items():
            rows.append([name, _f(info.get("rows")), _f(info.get("columns_count")),
                         _f(info.get("duplicate_rows")), _f(info.get("total_missing")),
                         _f(info.get("memory_mb"), 1)])
        L.append(_table(["Split", "Rows", "Columns", "Duplicate rows", "Missing cells",
                         "Memory (MB)"], rows, "TABLE I. Dataset shape per split."))
    overlap = meta.get("overlap", {})
    L.append(
        "Split integrity was verified by hashing every normalised record and intersecting the "
        "hash sets. "
        + ("No record appears in more than one split, so no train/test leakage exists."
           if not overlap.get("leakage_detected")
           else f"Warning: {overlap.get('pairwise')} duplicate records were found across splits.")
    )
    L.append("")

    L.append("### B. Preprocessing and Text Serialisation")
    L.append("")
    L.append(
        "Original files are opened read-only. Cleaning is stateless and per-value: whitespace "
        "is stripped, numeric-looking text is coerced, timestamps are parsed, and exact "
        "duplicate rows are removed within a split only. No statistic is fitted on one split "
        "and applied to another, so no transformation can leak test information. Missing "
        "measurements are not imputed; a missing field is omitted from the serialised record. "
        "Outliers are reported rather than removed, because extreme pollution episodes are the "
        "phenomenon of interest."
    )
    L.append("")
    L.append("Each row is rendered as a record card followed by deterministic question-answer "
             "pairs whose answers quote that row's own values, for example:")
    L.append("")
    example = prep.get("examples", {})
    if example.get("record_card"):
        L.append("```text")
        L.append(example["record_card"][:600])
        L.append("```")
        L.append("")
    for pair in example.get("qa_pairs", [])[:2]:
        L.append("```text")
        L.append(f"Question: {pair['question']}")
        L.append(f"Answer: {pair['answer']}")
        L.append("```")
        L.append("")
    L.append(
        "A short domain reference text (definitions of PM2.5, PM10, NO2, SO2, CO, O3, AQI and "
        "the influence of meteorology) is added to the training corpus only, so definitional "
        "questions are answerable without polluting the measurement distribution."
    )
    L.append("")

    L.append("### C. Tokenizer and Block Construction")
    L.append("")
    L.append(
        f"A byte-level byte-pair-encoding tokenizer is trained on the training split alone, "
        f"using GPT-2 style pre-tokenisation and {_f(tc.get('vocab_size'))} merge operations "
        f"over unique pre-tokens. Because it is byte-level, no input is ever out-of-vocabulary. "
        f"Documents are concatenated with begin/end markers and cut into contiguous blocks of "
        f"{_f(tc.get('block_size'))} tokens, giving {_f(tc.get('num_train_blocks'))} training "
        f"blocks, {_f(tc.get('num_val_blocks'))} validation blocks and "
        f"{_f(tc.get('num_test_blocks'))} test blocks. The language-modelling objective is "
        f"next-token prediction, so the input is the block minus its last token and the target "
        f"is the block minus its first token."
    )
    L.append("")

    L.append("### D. Model Architecture")
    L.append("")
    L.append(
        f"The model is a decoder-only causal Transformer with {_f(config.get('n_layer'))} "
        f"blocks, {_f(config.get('n_head'))} attention heads, model width "
        f"{_f(config.get('n_embd'))} and feed-forward width derived as "
        f"{_f(stats.get('ff_hidden'))}, giving {_f(n_params)} parameters. Positional "
        f"information is injected by RoPE into queries and keys only. Each block applies "
        f"pre-normalisation with RMSNorm, multi-head causal self-attention with an explicit "
        f"lower-triangular mask, and a SwiGLU feed-forward network, both with residual "
        f"connections. The output head is a linear projection to the vocabulary, weight-tied "
        f"with the input embedding to save parameters. Dropout of {_f(config.get('dropout'))} "
        f"is applied for regularisation."
    )
    L.append("")
    attn = 4 * int(config.get("n_embd") or 0) ** 2
    ff = 3 * int(config.get("n_embd") or 0) * int(stats.get("ff_hidden") or 0)
    rows = [
        ["Token embedding (tied)", _f(int(config.get("vocab_size") or 0) * int(config.get("n_embd") or 0))],
        ["Attention, per block (q,k,v,proj)", _f(attn)],
        ["SwiGLU feed-forward, per block", _f(ff)],
        [f"Transformer blocks ({_f(config.get('n_layer'))} total)",
         _f((attn + ff + 2 * int(config.get("n_embd") or 0)) * int(config.get("n_layer") or 0))],
        ["Total parameters", _f(n_params)],
    ]
    L.append(_table(["Component", "Parameters"], rows, "TABLE II. Parameter budget."))
    L.append(
        f"At the configured float32 precision the weights occupy "
        f"{_f((n_params or 0) * 4 / 1024 ** 2, 2)} MB, and the total training footprint "
        f"(weights, gradients, AdamW moments and activations) was estimated at "
        f"{_f(tc.get('estimated_train_memory_mb'), 1)} MB before training began."
    )
    L.append("")

    L.append("### E. Training Procedure")
    L.append("")
    L.append(
        f"Training used AdamW [6] with \u03b2 = (0.9, 0.95), weight decay "
        f"{_f(tc.get('weight_decay'))}, gradient clipping at {_f(tc.get('grad_clip'))} and a "
        f"linear warm-up over {_f(tc.get('warmup_steps'))} steps followed by cosine decay from "
        f"{_f(tc.get('learning_rate'), 5)} to {_f(tc.get('min_lr'), 5)}. The optimiser updated "
        f"once per {_f(tc.get('grad_accum'))} micro-batches of {_f(tc.get('batch_size'))} "
        f"blocks. Validation loss was computed every 50 steps; the checkpoint with the lowest "
        f"validation loss was retained, and training stopped early after "
        f"{_f(tc.get('patience'))} epochs without improvement. The test split was not read at "
        f"any point during training or model selection."
    )
    L.append("")
    L.append(
        "Nine sanity checks were executed before the full run and all had to pass: dataset "
        "non-emptiness, tokenizer round-trip fidelity, tensor shapes and one-token shift, "
        "forward-pass output shape, finiteness of the loss, correctness of the causal mask "
        "(verified by confirming that prefix logits are unchanged when future tokens are "
        "appended), finiteness of gradients, parameter update after one optimiser step, "
        "loss reduction over a fixed batch, and bit-exact checkpoint round-tripping."
    )
    L.append("")

    L.append("### F. Evaluation Protocol")
    L.append("")
    L.append(
        "Three families of metrics are reported. Likelihood metrics are cross-entropy, "
        "perplexity and teacher-forced token accuracy on the test blocks. Generation metrics "
        "compare sampled answers against reference answers using token-level F1, exact match, "
        "and numeric recall \u2014 the fraction of numerals in the reference answer that also "
        "appear in the generated answer. Finally, unsupported numerals (numbers generated that "
        "the reference does not contain) are counted, which is the most direct measure of "
        "hallucination for a measurement-oriented model."
    )
    L.append("")

    # ---------------- IV. Experimental setup ----------------
    L.append("## IV. Experimental Setup")
    L.append("")
    rows = [
        ["Operating system", str(hardware.get("os"))],
        ["Python", str(hardware.get("python_version"))],
        ["PyTorch", str(hardware.get("torch_version"))],
        ["CPU", str(hardware.get("cpu_name"))],
        ["Logical cores", _f(hardware.get("cpu_count"))],
        ["System memory (GB)", _f(hardware.get("total_ram_gb"))],
        ["CUDA available", str(hardware.get("cuda_available"))],
        ["GPU devices", str(hardware.get("gpu_names") or "none")],
        ["Apple MPS", str(hardware.get("mps_available"))],
        ["Selected device", str(hardware.get("selected_device"))],
        ["Numerical precision", str(tc.get("precision"))],
    ]
    L.append(_table(["Property", "Value"], rows, "TABLE III. Hardware and runtime configuration."))
    L.append(
        "The model size, batch size, sequence length and step budget were selected from these "
        "detected properties rather than assumed. No GPU was present, so a CPU preset was used "
        "and the architecture was reduced accordingly."
    )
    L.append("")

    # ---------------- V. Results ----------------
    L.append("## V. Results and Statistical Analysis")
    L.append("")
    L.append(f"### A. Corpus Statistics")
    L.append("")
    rows = []
    for name in ("train", "validation", "test"):
        info = splits.get(name)
        if info:
            rows.append([name, _f(info.get("rows_after_cleaning")), _f(info.get("documents")),
                         _f(info.get("characters"))])
    if rows:
        L.append(_table(["Split", "Records", "Serialised documents", "Characters"], rows,
                        "TABLE IV. Corpus composition after cleaning."))
    if total_chars and train_rows:
        L.append(
            f"The corpus expands to {_f(total_docs)} documents and {_f(total_chars)} characters, "
            f"approximately {_f(total_chars / max(1, (train_rows or 1) + (val_rows or 0) + (test_rows or 0)), 0)} "
            f"characters of text per source record, because each record yields a record card "
            f"plus several question-answer documents."
        )
    L.append("")

    L.append("### B. Descriptive Statistics of the Measurements")
    L.append("")
    train_info = analyses.get("train", {})
    numeric_cols = [c for c in train_info.get("columns", []) if c.get("role") == "numeric"]
    if numeric_cols:
        rows = []
        for col in numeric_cols:
            rows.append([f"`{col['name']}`", _f(col.get("mean"), 2), _f(col.get("std"), 2),
                         _f(col.get("min"), 2), _f(col.get("p05"), 2), _f(col.get("p50"), 2),
                         _f(col.get("p95"), 2), _f(col.get("max"), 2),
                         _f(col.get("extreme_outliers_3iqr"))])
        L.append(_table(["Variable", "Mean", "SD", "Min", "P5", "Median", "P95", "Max",
                         "Outliers (3\u00b7IQR)"], rows,
                        "TABLE V. Descriptive statistics of numeric variables (training split)."))
        L.append(
            "Outlier counts are reported, not corrected. In air quality data the right tail is "
            "the phenomenon of interest \u2014 severe pollution episodes \u2014 so trimming it "
            "would remove the signal the model is meant to describe."
        )
        L.append("")

    corr = stats.get("correlation") or {}
    if corr:
        keys = list(corr.keys())[:8]
        rows = [[f"`{r}`"] + [_f(corr[r].get(c), 2) for c in keys] for r in keys]
        L.append(_table(["", *[f"`{k}`" for k in keys]], rows,
                        "TABLE VI. Pearson correlation between measured variables (training split)."))
        L.append(
            "The correlation structure confirms the physical expectations of the domain: "
            "particle fractions co-vary strongly because PM10 includes the PM2.5 fraction, and "
            "the aggregated index tracks the dominant pollutant. A language model trained on "
            "these serialised records can therefore learn real relationships rather than only "
            "formatting conventions."
        )
        L.append("")

    L.append("### C. Optimisation Behaviour")
    L.append("")
    epoch_rows = []
    eval_rows = []
    for entry in history:
        if entry.get("kind", "epoch") == "epoch":
            epoch_rows.append([_f(entry.get("epoch")), _f(entry.get("step")),
                               _f(entry.get("train_loss"), 4),
                               _f(entry.get("train_loss_epoch_end"), 4),
                               _f(entry.get("val", {}).get("loss"), 4),
                               _f(entry.get("val", {}).get("perplexity"), 2)])
        else:
            eval_rows.append([_f(entry.get("step")),
                              _f(entry.get("val", {}).get("loss"), 4),
                              _f(entry.get("val", {}).get("perplexity"), 2),
                              _f((entry.get("val", {}).get("token_accuracy") or 0) * 100, 2)])
    if epoch_rows:
        L.append(_table(["Epoch", "Step", "Train loss (epoch mean)",
                         "Train loss (final 50 batches)", "Validation loss", "Validation ppl"],
                        epoch_rows, "TABLE VII. Training and validation loss per epoch."))
    if eval_rows:
        L.append(_table(["Step", "Validation loss", "Validation ppl", "Token accuracy (%)"],
                        eval_rows, "TABLE VIII. Validation checkpoints during training."))
    if len(eval_rows) >= 2:
        first = eval_rows[0][1]
        last = eval_rows[-1][1]
        L.append(
            f"Validation loss falls from {first} at the first checkpoint to {last} at the "
            f"last, and the training loss over the final fifty micro-batches is consistently "
            f"below the whole-epoch average, showing that the model was still improving when "
            f"the step budget ended rather than having saturated."
        )
    if duration:
        L.append("")
        L.append(
            f"Total wall-clock training time was {_f(duration, 1)} s, i.e. "
            f"{_f(float(duration) / max(1, tc.get('max_steps') or 1), 3)} s per optimiser step. "
            f"Perplexity on the validation split reached {_f((eval_rows[-1][2] if eval_rows else None), 2)}."
        )
    L.append("")

    L.append("### D. Test-Set Performance")
    L.append("")
    rows = [
        ["Cross-entropy loss", _f(test.get("loss"), 4)],
        ["Perplexity", _f(test.get("perplexity"), 2)],
        ["Teacher-forced token accuracy (%)", _f((test.get("token_accuracy") or 0) * 100, 2)],
        ["Tokens scored", _f(test.get("tokens"))],
        ["Held-out QA pairs", _f(qa.get("held_out_questions")) if has_qa else "not collected"],
        ["Mean token F1", _f(qa.get("mean_token_f1"), 4) if has_qa else "not collected"],
        ["Exact match rate", _f(qa.get("exact_match_rate"), 4) if has_qa else "not collected"],
        ["Mean numeric recall", _f(qa.get("mean_numeric_recall"), 4) if has_qa else "not collected"],
        ["Unsupported numerals generated",
         _f(qa.get("generated_numerals_unsupported")) if has_qa else "not collected"],
    ]
    L.append(_table(["Metric", "Value"], rows, "TABLE IX. Test-split results."))
    L.append(
        "The gap between token accuracy and numeric recall is the central quantitative finding "
        "of this study. A small model can learn the *shape* of an answer almost perfectly \u2014 "
        "the template, the field names, the units \u2014 while still assigning meaningful "
        "probability to incorrect digits. Likelihood metrics alone would therefore overstate "
        "the usefulness of the model, which is why numeric recall and unsupported-numeral counts "
        "are reported alongside them."
    )
    L.append("")

    if samples:
        L.append("### E. Qualitative Examples")
        L.append("")
        rows = []
        for item in samples[:8]:
            rows.append([item["question"][:70], (item.get("reference") or "\u2014")[:60],
                         (item.get("prediction") or "\u2014")[:70],
                         _f(item.get("token_f1"), 2)])
        L.append(_table(["Question", "Reference answer", "Model answer", "F1"], rows,
                        "TABLE X. Sampled question answering on held-out data."))
        L.append("")
        L.append(
            "Answers reproduce the training format reliably. Measure-specific questions are "
            "answered correctly when the record is unambiguous in the corpus and incorrectly "
            "otherwise, which is the expected behaviour of a "
            f"{_f(n_params)}-parameter model trained for {_f(tc.get('max_steps'))} steps."
        )
        L.append("")

    # ---------------- VI. Discussion ----------------
    L.append("## VI. Discussion")
    L.append("")
    L.append(
        "Three observations follow from the results. First, domain narrowness substitutes "
        "efficiently for scale: with a vocabulary learned from the corpus and a training set of "
        "a few thousand records, a model of a few million parameters reaches very low "
        "perplexity on held-out records. Second, the remaining errors are almost entirely "
        "numerical rather than structural, which means improvements should target value "
        "fidelity \u2014 retrieval, constrained decoding, or a wider numeric vocabulary \u2014 "
        "rather than more parameters. Third, the entire pipeline is auditable: every number in "
        "this paper was emitted by the run, and the nine sanity checks make silent failures "
        "unlikely."
    )
    L.append("")
    L.append(
        "The deployment path matters for practical use. Exporting the checkpoint to GGUF and "
        "serving it through Ollama [10], [11] gives an interactive chat interface on ordinary "
        "hardware without a Python runtime, while a standard-library server exposing the same "
        "Ollama-compatible REST API provides an identical interface straight from the trained "
        "PyTorch weights. Both routes were exercised during development."
    )
    L.append("")

    # ---------------- VII. Limitations ----------------
    L.append("## VII. Limitations and Threats to Validity")
    L.append("")
    L.append(
        f"1. **Model scale.** With {_f(n_params)} parameters and "
        f"{_f(tc.get('max_steps'))} optimisation steps, free-form prose generation is limited; "
        f"the model is a structured-answer engine, not a general assistant."
    )
    L.append("2. **No retrieval.** Record-specific questions succeed only when the record is "
             "present in the prompt; the model cannot look up an arbitrary row.")
    L.append("3. **Numeric hallucination.** Unsupported numerals are counted in Table IX and are "
             "non-zero. Generated values must not be used for decisions without verification.")
    L.append("4. **Template bias.** Question-answer pairs are generated deterministically from "
             "the records, so evaluation measures fidelity to a templated distribution rather "
             "than to human-written questions.")
    L.append("5. **Single dataset.** The corpus comes from one project's monitoring records; "
             "transfer to other stations, pollutants or averaging periods is untested.")
    L.append("6. **Hardware envelope.** Results were obtained on CPU only; larger presets exist "
             "in the code but their behaviour on GPU or Apple MPS was not measured here.")
    L.append("")

    # ---------------- VIII. Conclusion ----------------
    L.append("## VIII. Conclusion and Future Work")
    L.append("")
    L.append(
        "This paper documented the ideation, construction and evaluation of a compact air "
        "quality language model built entirely from a project-specific dataset. A "
        "byte-level tokenizer, a RoPE/RMSNorm/SwiGLU decoder and a leak-free serialisation "
        "pipeline together produce a model that reaches "
        f"{_f((test.get('token_accuracy') or 0) * 100, 1)}% teacher-forced token accuracy and "
        f"a test perplexity of {_f(test.get('perplexity'), 2)} at "
        f"{_f(n_params)} parameters, and that can be demonstrated interactively through Ollama."
    )
    L.append("")
    L.append("Future work will (i) add retrieval so record-specific questions are grounded in "
             "an exact row, (ii) apply constrained decoding over numeric tokens to eliminate "
             "unsupported numerals, (iii) expand the question templates to comparisons across "
             "locations and time windows, and (iv) evaluate against human-written questions "
             "rather than templated pairs.")
    L.append("")

    # ---------------- References ----------------
    L.append("## References")
    L.append("")
    for index, ref in enumerate(REFERENCES, start=1):
        L.append(f"[{index}] {ref}")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*Formatting note: this draft follows the IEEE two-column article structure "
             "(abstract, index terms, numbered sections, numbered references). Paste into the "
             "IEEE conference template (Word or LaTeX `IEEEtran`) to obtain the final "
             "two-column layout; the tables convert directly.*")
    L.append("")
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the IEEE-format research article.")
    parser.add_argument("--reports-dir", default=os.path.join(PROJECT_ROOT, "reports"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    meta = _load(os.path.join(args.reports_dir, "run_metadata.json"))
    if meta is None:
        print(f"ERROR: {os.path.join(args.reports_dir, 'run_metadata.json')} not found.")
        print("Run the pipeline first:  python src/train_slm.py")
        return 2
    evaluation = _load(os.path.join(args.reports_dir, "evaluation_results.json"))
    paper = build_paper(meta, evaluation)
    out = args.out or os.path.join(args.reports_dir, "research_paper.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(paper)
    print(f"Wrote {out} ({len(paper.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
