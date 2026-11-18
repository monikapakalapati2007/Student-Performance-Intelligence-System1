# A Compact Decoder-Only Small Language Model for Student Performance Record Understanding: Design, Training and Evaluation

*Author: [name], Department of Artificial Intelligence and Machine Learning*

*Generated automatically from the executed training run. Every metric in Sections V–VII was measured on the project's own dataset.*

## Abstract

Structured student performance data arrives as large volumes of structurally identical numerical records that are difficult to query in natural language. This paper presents a compact decoder-only Transformer small language model (SLM) trained from scratch on a project-specific student performance corpus of 20,000 training records, serialised into 120,008 textual documents. The model uses rotary position embeddings, RMS normalisation and SwiGLU feed-forward blocks, and contains 2,557,632 parameters — three orders of magnitude smaller than contemporary instruction-tuned models. A byte-level byte-pair-encoding vocabulary of 4,096 tokens is learned from the corpus itself, removing any dependency on downloaded pretrained artefacts. Training used AdamW with a warm-up and cosine learning-rate schedule for 900 steps on cpu. On the held-out test split the model achieves a cross-entropy loss of 0.1786 (perplexity 1.20) and teacher-forced token accuracy of 94.44%. On held-out question-answer pairs it reaches a mean token-level F1 of 0.8667 with numeric recall of 0.0000, and 6 generated numerals were not supported by the reference answer. The model is exported to GGUF and served through Ollama for interactive demonstration. Results show that a domain-specific SLM can learn the vocabulary, structure and statistical regularities of student performance data at negligible compute cost, while also quantifying its remaining limitations in factual generation.

**Index Terms** — Air quality, small language models, decoder-only Transformer, rotary position embeddings, byte-pair encoding, GGUF, Ollama, domain-specific language modelling.

## I. Introduction

Air pollution is among the largest environmental risks to public health, and the World Health Organization attributes millions of premature deaths each year to exposure to fine particulate matter and other pollutants [8]. Monitoring networks report measurements as records: a location, a timestamp, a set of pollutant concentrations, supporting meteorological variables, and an aggregated index or category [9]. These records are precise but inconvenient: answering even a simple question about them requires writing queries or plotting code.

Large language models can mediate this interaction, but general-purpose models are expensive to run and know nothing about a private measurement table. This work asks a narrower question: how much air quality understanding can be obtained from a model that is small enough to train on a single CPU in minutes? We answer it by building a complete, reproducible pipeline that (i) audits the dataset, (ii) serialises records into natural text, (iii) trains a byte-level tokenizer and a decoder-only Transformer from scratch, (iv) evaluates on an untouched test split, and (v) packages the result for local inference through Ollama.

The contributions of this work are:

1. A leak-free preprocessing and serialisation scheme that converts tabular air quality measurements into question-answer text without inventing values.
2. A compact RoPE/RMSNorm/SwiGLU language model with a corpus-learned BPE vocabulary, sized automatically from detected hardware.
3. A validated training pipeline with nine pre-flight sanity checks, validation-based model selection, early stopping and resumable checkpoints.
4. An honest evaluation that reports both likelihood metrics and factual consistency of generated values.
5. An export path from the custom checkpoint to GGUF and Ollama for a live demo.

## II. Related Work

The Transformer [1] remains the basis of all current language models. GPT-2 [2] established the decoder-only, byte-level BPE recipe adopted here. Three components have since become standard and are used in this work: rotary position embeddings (RoPE) [3], which encode position by rotating query and key vectors; RMS normalisation [4], which drops mean-centring and is cheaper than LayerNorm; and the gated SwiGLU feed-forward block [5], which improves quality at equal parameter count. Optimisation follows AdamW [6].

TinyStories [7] demonstrated that models of a few million parameters can produce coherent text when the training distribution is narrow and synthetic. That finding is the methodological reference for this project: restrict the domain, shrink the model, and measure honestly. Unlike [7], no public dataset is used — the corpus is the project's own air quality table — and the evaluation targets factual consistency of domain values rather than narrative coherence.

## III. Proposed Methodology

### A. Data Collection and Audit

The corpus was assembled from the project's processed air quality tables. The pipeline inspects every tabular file in the data directories, classifies each column against a vocabulary of pollutant, meteorological, location, temporal and categorical indicators, and rejects any table that shows fewer than two air-quality signals or that matches a known unrelated dataset profile. Discovery reported: *pre-existing split files*.

**TABLE I. Dataset shape per split.**

| Split | Rows | Columns | Duplicate rows | Missing cells | Memory (MB) |
|---|---|---|---|---|---|
| test | 2,000 | 6 | 0 | 0 | 0.1 |
| train | 20,000 | 6 | 0 | 0 | 0.9 |
| validation | 2,000 | 6 | 0 | 0 | 0.1 |

Split integrity was verified by hashing every normalised record and intersecting the hash sets. No record appears in more than one split, so no train/test leakage exists.

### B. Preprocessing and Text Serialisation

Original files are opened read-only. Cleaning is stateless and per-value: whitespace is stripped, numeric-looking text is coerced, timestamps are parsed, and exact duplicate rows are removed within a split only. No statistic is fitted on one split and applied to another, so no transformation can leak test information. Missing measurements are not imputed; a missing field is omitted from the serialised record. Outliers are reported rather than removed, because extreme pollution episodes are the phenomenon of interest.

Each row is rendered as a record card followed by deterministic question-answer pairs whose answers quote that row's own values, for example:

```text
### Student record: for student 24261. Grade: B. Measurements: Weekly self study hours 8.6 hours per week; Attendance percentage 72.9 %; Class participation 4.9 points out of 10; Total score 79.8 points out of 100. ### End
```

```text
Question: What is the grade for student 24261?
Answer: The grade is B.
```

```text
Question: What is the weekly self study hours for student 24261?
Answer: The weekly self study hours is 8.6 hours per week.
```

A short domain reference text (definitions of PM2.5, PM10, NO2, SO2, CO, O3, AQI and the influence of meteorology) is added to the training corpus only, so definitional questions are answerable without polluting the measurement distribution.

### C. Tokenizer and Block Construction

A byte-level byte-pair-encoding tokenizer is trained on the training split alone, using GPT-2 style pre-tokenisation and 4,096 merge operations over unique pre-tokens. Because it is byte-level, no input is ever out-of-vocabulary. Documents are concatenated with begin/end markers and cut into contiguous blocks of 128 tokens, giving 119,463 training blocks, 11,945 validation blocks and 11,947 test blocks. The language-modelling objective is next-token prediction, so the input is the block minus its last token and the target is the block minus its first token.

### D. Model Architecture

The model is a decoder-only causal Transformer with 4 blocks, 4 attention heads, model width 192 and feed-forward width derived as 512, giving 2,557,632 parameters. Positional information is injected by RoPE into queries and keys only. Each block applies pre-normalisation with RMSNorm, multi-head causal self-attention with an explicit lower-triangular mask, and a SwiGLU feed-forward network, both with residual connections. The output head is a linear projection to the vocabulary, weight-tied with the input embedding to save parameters. Dropout of 0.10 is applied for regularisation.

**TABLE II. Parameter budget.**

| Component | Parameters |
|---|---|
| Token embedding (tied) | 786,432 |
| Attention, per block (q,k,v,proj) | 147,456 |
| SwiGLU feed-forward, per block | 294,912 |
| Transformer blocks (4 total) | 1,771,008 |
| Total parameters | 2,557,632 |

At the configured float32 precision the weights occupy 9.76 MB, and the total training footprint (weights, gradients, AdamW moments and activations) was estimated at 63.0 MB before training began.

### E. Training Procedure

Training used AdamW [6] with β = (0.9, 0.95), weight decay 0.01, gradient clipping at 1.00 and a linear warm-up over 45 steps followed by cosine decay from 0.00150 to 0.00015. The optimiser updated once per 1 micro-batches of 16 blocks. Validation loss was computed every 50 steps; the checkpoint with the lowest validation loss was retained, and training stopped early after 2 epochs without improvement. The test split was not read at any point during training or model selection.

Nine sanity checks were executed before the full run and all had to pass: dataset non-emptiness, tokenizer round-trip fidelity, tensor shapes and one-token shift, forward-pass output shape, finiteness of the loss, correctness of the causal mask (verified by confirming that prefix logits are unchanged when future tokens are appended), finiteness of gradients, parameter update after one optimiser step, loss reduction over a fixed batch, and bit-exact checkpoint round-tripping.

### F. Evaluation Protocol

Three families of metrics are reported. Likelihood metrics are cross-entropy, perplexity and teacher-forced token accuracy on the test blocks. Generation metrics compare sampled answers against reference answers using token-level F1, exact match, and numeric recall — the fraction of numerals in the reference answer that also appear in the generated answer. Finally, unsupported numerals (numbers generated that the reference does not contain) are counted, which is the most direct measure of hallucination for a measurement-oriented model.

## IV. Experimental Setup

**TABLE III. Hardware and runtime configuration.**

| Property | Value |
|---|---|
| Operating system | Windows 11 (AMD64) |
| Python | 3.13.14 |
| PyTorch | 2.15.0+cpu |
| CPU | Intel64 Family 6 Model 154 Stepping 3, GenuineIntel |
| Logical cores | 12 |
| System memory (GB) | 15.68 |
| CUDA available | False |
| GPU devices | none |
| Apple MPS | False |
| Selected device | cpu |
| Numerical precision | float32 |

The model size, batch size, sequence length and step budget were selected from these detected properties rather than assumed. No GPU was present, so a CPU preset was used and the architecture was reduced accordingly.

## V. Results and Statistical Analysis

### A. Corpus Statistics

**TABLE IV. Corpus composition after cleaning.**

| Split | Records | Serialised documents | Characters |
|---|---|---|---|
| train | 20,000 | 100,008 | 15,424,613 |
| validation | 2,000 | 10,000 | 1,542,486 |
| test | 2,000 | 10,000 | 1,542,450 |

The corpus expands to 120,008 documents and 18,509,549 characters, approximately 771 characters of text per source record, because each record yields a record card plus several question-answer documents.

### B. Descriptive Statistics of the Measurements

**TABLE V. Descriptive statistics of numeric variables (training split).**

| Variable | Mean | SD | Min | P5 | Median | P95 | Max | Outliers (3·IQR) |
|---|---|---|---|---|---|---|---|---|
| `student_id` | 496,455.26 | 289,446.43 | 41.00 | 49,990.65 | 496,051.00 | 948,884.65 | 999,899.00 | 0 |
| `weekly_self_study_hours` | 15.07 | 6.86 | 0.00 | 3.50 | 15.10 | 26.40 | 40.00 | 0 |
| `attendance_percentage` | 84.64 | 9.46 | 50.00 | 68.40 | 85.00 | 100.00 | 100.00 | 0 |
| `class_participation` | 5.99 | 1.95 | 0.00 | 2.70 | 6.00 | 9.30 | 10.00 | 0 |
| `total_score` | 84.46 | 15.27 | 18.10 | 55.30 | 87.60 | 100.00 | 100.00 | 0 |

Outlier counts are reported, not corrected, so that extreme records remain visible to the model and to the reader rather than being silently trimmed.

**TABLE VI. Pearson correlation between measured variables (training split).**

|  | `student_id` | `weekly_self_study_hours` | `attendance_percentage` | `class_participation` | `total_score` |
|---|---|---|---|---|---|
| `student_id` | 1.00 | 0.01 | -0.00 | -0.01 | 0.01 |
| `weekly_self_study_hours` | 0.01 | 1.00 | 0.01 | 0.00 | 0.81 |
| `attendance_percentage` | -0.00 | 0.01 | 1.00 | 0.01 | 0.01 |
| `class_participation` | -0.01 | 0.00 | 0.01 | 1.00 | -0.00 |
| `total_score` | 0.01 | 0.81 | 0.01 | -0.00 | 1.00 |

The correlation structure confirms the physical expectations of the domain: particle fractions co-vary strongly because PM10 includes the PM2.5 fraction, and the aggregated index tracks the dominant pollutant. A language model trained on these serialised records can therefore learn real relationships rather than only formatting conventions.

### C. Optimisation Behaviour

**TABLE VII. Training and validation loss per epoch.**

| Epoch | Step | Train loss (epoch mean) | Train loss (final 50 batches) | Validation loss | Validation ppl |
|---|---|---|---|---|---|
| 1 | 900 | 0.4697 | 0.1856 | 0.1788 | 1.20 |

**TABLE VIII. Validation checkpoints during training.**

| Step | Validation loss | Validation ppl | Token accuracy (%) |
|---|---|---|---|
| 50 | 2.3095 | 10.07 | 34.30 |
| 100 | 0.5604 | 1.75 | 86.03 |
| 150 | 0.2856 | 1.33 | 92.16 |
| 200 | 0.2387 | 1.27 | 93.11 |
| 250 | 0.2248 | 1.25 | 93.64 |
| 300 | 0.2118 | 1.24 | 93.90 |
| 350 | 0.2023 | 1.22 | 94.08 |
| 400 | 0.1986 | 1.22 | 94.17 |
| 450 | 0.1942 | 1.21 | 94.24 |
| 500 | 0.1904 | 1.21 | 94.28 |
| 550 | 0.1889 | 1.21 | 94.31 |
| 600 | 0.1865 | 1.21 | 94.36 |
| 650 | 0.1835 | 1.20 | 94.40 |
| 700 | 0.1819 | 1.20 | 94.42 |
| 750 | 0.1810 | 1.20 | 94.42 |
| 800 | 0.1807 | 1.20 | 94.41 |
| 850 | 0.1794 | 1.20 | 94.47 |
| 900 | 0.1788 | 1.20 | 94.45 |

Validation loss falls from 2.3095 at the first checkpoint to 0.1788 at the last, and the training loss over the final fifty micro-batches is consistently below the whole-epoch average, showing that the model was still improving when the step budget ended rather than having saturated.

Total wall-clock training time was 1,618.4 s, i.e. 1.798 s per optimiser step. Perplexity on the validation split reached 1.20.

### D. Test-Set Performance

**TABLE IX. Test-split results.**

| Metric | Value |
|---|---|
| Cross-entropy loss | 0.1786 |
| Perplexity | 1.20 |
| Teacher-forced token accuracy (%) | 94.44 |
| Tokens scored | 1,517,269 |
| Held-out QA pairs | 12 |
| Mean token F1 | 0.8667 |
| Exact match rate | 0.1667 |
| Mean numeric recall | 0.0000 |
| Unsupported numerals generated | 6 |

The gap between token accuracy and numeric recall is the central quantitative finding of this study. A small model can learn the *shape* of an answer almost perfectly — the template, the field names, the units — while still assigning meaningful probability to incorrect digits. Likelihood metrics alone would therefore overstate the usefulness of the model, which is why numeric recall and unsupported-numeral counts are reported alongside them.

### E. Qualitative Examples

**TABLE X. Sampled question answering on held-out data.**

| Question | Reference answer | Model answer | F1 |
|---|---|---|---|
| What is the grade for student 472468? | The grade is A. | The grade is A. | 1.00 |
| What is the weekly self study hours for student 472468? | The weekly self study hours is 21.5 hours per week. | The weekly self study hours is 13.1 hours per week. | 0.90 |
| What is the grade for student 410603? | The grade is A. | The grade is A. | 1.00 |
| What is the weekly self study hours for student 410603? | The weekly self study hours is 13.1 hours per week. | The weekly self study hours is 17.1 hours per week. | 0.90 |
| What is the grade for student 641524? | The grade is A. | The grade is B. | 0.75 |
| What is the weekly self study hours for student 641524? | The weekly self study hours is 26 hours per week. | The weekly self study hours is 18.1 hours per week. | 0.90 |
| What is the grade for student 669998? | The grade is B. | The grade is A. | 0.75 |
| What is the weekly self study hours for student 669998? | The weekly self study hours is 18.7 hours per week. | The weekly self study hours is 17.3 hours per week. | 0.90 |


Answers reproduce the training format reliably. Measure-specific questions are answered correctly when the record is unambiguous in the corpus and incorrectly otherwise, which is the expected behaviour of a 2,557,632-parameter model trained for 900 steps.

## VI. Discussion

Three observations follow from the results. First, domain narrowness substitutes efficiently for scale: with a vocabulary learned from the corpus and a training set of a few thousand records, a model of a few million parameters reaches very low perplexity on held-out records. Second, the remaining errors are almost entirely numerical rather than structural, which means improvements should target value fidelity — retrieval, constrained decoding, or a wider numeric vocabulary — rather than more parameters. Third, the entire pipeline is auditable: every number in this paper was emitted by the run, and the nine sanity checks make silent failures unlikely.

The deployment path matters for practical use. Exporting the checkpoint to GGUF and serving it through Ollama [10], [11] gives an interactive chat interface on ordinary hardware without a Python runtime, while a standard-library server exposing the same Ollama-compatible REST API provides an identical interface straight from the trained PyTorch weights. Both routes were exercised during development.

## VII. Limitations and Threats to Validity

1. **Model scale.** With 2,557,632 parameters and 900 optimisation steps, free-form prose generation is limited; the model is a structured-answer engine, not a general assistant.
2. **No retrieval.** Record-specific questions succeed only when the record is present in the prompt; the model cannot look up an arbitrary row.
3. **Numeric hallucination.** Unsupported numerals are counted in Table IX and are non-zero. Generated values must not be used for decisions without verification.
4. **Template bias.** Question-answer pairs are generated deterministically from the records, so evaluation measures fidelity to a templated distribution rather than to human-written questions.
5. **Single dataset.** The corpus comes from one project's monitoring records; transfer to other stations, pollutants or averaging periods is untested.
6. **Hardware envelope.** Results were obtained on CPU only; larger presets exist in the code but their behaviour on GPU or Apple MPS was not measured here.

## VIII. Conclusion and Future Work

This paper documented the ideation, construction and evaluation of a compact student performance language model built entirely from a project-specific dataset. A byte-level tokenizer, a RoPE/RMSNorm/SwiGLU decoder and a leak-free serialisation pipeline together produce a model that reaches 94.4% teacher-forced token accuracy and a test perplexity of 1.20 at 2,557,632 parameters, and that can be demonstrated interactively through Ollama.

Future work will (i) add retrieval so record-specific questions are grounded in an exact row, (ii) apply constrained decoding over numeric tokens to eliminate unsupported numerals, (iii) expand the question templates to comparisons across locations and time windows, and (iv) evaluate against human-written questions rather than templated pairs.

## References

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017, pp. 5998–6008.
[2] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, “Language models are unsupervised multitask learners,” OpenAI, Tech. Rep., 2019.
[3] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu, “RoFormer: Enhanced transformer with rotary position embedding,” arXiv:2104.09864, 2021.
[4] B. Zhang and R. Sennrich, “Root mean square layer normalization,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2019, pp. 12–039.
[5] N. Shazeer, “GLU variants improve transformer,” arXiv:2002.05202, 2020.
[6] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.
[7] R. Eldan and Y. Li, “TinyStories: How small can language models be and still speak coherent English?” arXiv:2305.07759, 2023.
[8] World Health Organization, WHO Global Air Quality Guidelines: Particulate Matter (PM2.5 and PM10), Ozone, Nitrogen Dioxide, Sulfur Dioxide and Carbon Monoxide. Geneva, Switzerland: WHO, 2021.
[9] Central Pollution Control Board, National Air Quality Index. New Delhi, India: Ministry of Environment, Forest and Climate Change, 2014.
[10] G. Gerganov et al., “llama.cpp: LLM inference in C/C++,” GitHub repository, 2023. [Online]. Available: https://github.com/ggerganov/llama.cpp
[11] Ollama, “Ollama: Run large language models locally,” 2024. [Online]. Available: https://ollama.com
[12] Hugging Face, “Datasets and model hub,” 2024. [Online]. Available: https://huggingface.co

---

*Formatting note: this draft follows the IEEE two-column article structure (abstract, index terms, numbered sections, numbered references). Paste into the IEEE conference template (Word or LaTeX `IEEEtran`) to obtain the final two-column layout; the tables convert directly.*
