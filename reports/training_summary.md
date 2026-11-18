# Student Performance SLM — Training Summary

## 1. Project description

A compact decoder-only causal Transformer language model (≈2–6M parameters) trained from scratch on structured student performance records. Each record is serialised into readable text together with deterministic question/answer pairs derived only from the fields present in the row. The model therefore learns the vocabulary and relationships of the dataset rather than memorising a general-purpose web corpus. No pretrained weights are used.

## 2. Dataset information

- Dataset profile: **student_performance** (student performance)
- Discovery: pre-existing split files
- Rows used: {'test': 2000, 'train': 20000, 'validation': 2000} of {'test': 150000, 'train': 700000, 'validation': 150000} (seeded subsample, seed 1337)
| split | rows | columns | duplicate rows | rows used | documents |
|---|---|---|---|---|---|
| test | 2,000 | 6 | 0 | 2,000 | 10,000 |
| train | 20,000 | 6 | 0 | 20,000 | 100,008 |
| validation | 2,000 | 6 | 0 | 2,000 | 10,000 |

Total serialised text: **18,509,549 characters** across **120,008 documents**.

## 3. Preprocessing methodology

1. Every table in the project data directories is inspected and classified; non-air-quality tables (for example student-performance data) are rejected.
2. Existing train/validation/test files are used exactly as provided. If no split files exist, one seeded 80/10/10 split is created **once** and frozen on disk.
3. Values are whitespace-stripped, numeric-looking text is coerced, timestamps are parsed, and exact duplicates are removed *within* each split only.
4. Each row becomes a record card plus deterministic QA pairs whose answers quote the row's own values. Missing fields are omitted, never imputed.
5. Domain reference text (pollutant definitions and general air quality knowledge) is added to the training corpus only, so the model can answer definitional questions.
6. The test split is written to disk and never seen during training or model selection.

## 4. Split sizes

- **train**: 20,000 rows → 100,008 documents
- **validation**: 2,000 rows → 10,000 documents
- **test**: 2,000 rows → 10,000 documents

## 5. Tokenizer

- Type: byte-level BPE trained from scratch on the training corpus
- Vocabulary size: **4,096** tokens
- Training blocks: 119,463 × 128 tokens
- Validation blocks: 11,945 | Test blocks: 11,947
- Saved to: `C:\Users\gnaya\OneDrive\Desktop\design thinking\models\student_performance_slm\tokenizer`

## 6. Model architecture

- Architecture: **decoder_only_rmsnorm_rope_swiglu**
- Layers: 4 | Heads: 4 | d_model: 192 | head_dim: 48
- Context length: 128 tokens
- Positional encoding: **Rotary Position Embeddings (RoPE)**
- Normalisation: **RMSNorm** (pre-norm)
- Feed-forward: **SwiGLU** gated MLP
- Attention: multi-head causal self-attention with an explicit lower-triangular mask
- Output head: linear projection to vocabulary, weight-tied with the token embedding = True
- Dropout: 0.1

**Total parameters: 2,557,632**

## 7. Hardware configuration

- OS: Windows 11 (AMD64)
- Python: 3.13.14
- PyTorch: 2.15.0+cpu
- CPU: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- Logical cores: 12
- System RAM (GB): 15.68
- CUDA available: False
- GPU: []
- GPU VRAM total (GB): None
- GPU VRAM free (GB): None
- Apple MPS: False
- Selected device: cpu

Estimated training memory: 63.0 MB

## 8. Training hyperparameters

- preset: cpu_tiny
- optimizer: AdamW
- schedule: linear warmup + cosine decay
- learning_rate: 0.0015
- min_lr: 0.00015000000000000001
- warmup_steps: 45
- weight_decay: 0.01
- batch_size: 16
- grad_accum: 1
- block_size: 128
- max_steps: 900
- max_epochs: 4
- grad_clip: 1.0
- precision: float32
- seed: 1337
- patience: 2

## 9. Epochs and duration

- Epochs requested: 4 | steps executed: 900 maximum
- Wall-clock training time: **1618.4 s**
- Stage timings (s): hardware=0.0, preprocess=5.7, sanity=124.3, train=1808.1, total=1863.5

## 10. Loss per epoch

| epoch | step | train loss (epoch mean) | train loss (last 50 steps) | train tok acc | val loss | val ppl | val tok acc |
|---|---|---|---|---|---|---|---|
| 1 | 900 | 0.4697 | 0.1856 | 0.8726 | 0.1788 | 1.1958 | 0.9445 |

_Training loss is also reported as the mean of the final 50 micro-batches of the epoch, which reflects the model's state at the end of that epoch rather than an average diluted by the untrained starting weights._

### Validation checkpoints during training

| step | val loss | val ppl | val tok acc |
|---|---|---|---|
| 50 | 2.3095 | 10.0696 | 0.3430 |
| 100 | 0.5604 | 1.7514 | 0.8603 |
| 150 | 0.2856 | 1.3306 | 0.9216 |
| 200 | 0.2387 | 1.2697 | 0.9311 |
| 250 | 0.2248 | 1.2520 | 0.9364 |
| 300 | 0.2118 | 1.2359 | 0.9390 |
| 350 | 0.2023 | 1.2242 | 0.9408 |
| 400 | 0.1986 | 1.2196 | 0.9417 |
| 450 | 0.1942 | 1.2144 | 0.9424 |
| 500 | 0.1904 | 1.2098 | 0.9428 |
| 550 | 0.1889 | 1.2079 | 0.9431 |
| 600 | 0.1865 | 1.2050 | 0.9436 |
| 650 | 0.1835 | 1.2015 | 0.9440 |
| 700 | 0.1819 | 1.1995 | 0.9442 |
| 750 | 0.1810 | 1.1984 | 0.9442 |
| 800 | 0.1807 | 1.1981 | 0.9441 |
| 850 | 0.1794 | 1.1965 | 0.9447 |
| 900 | 0.1788 | 1.1958 | 0.9445 |

## 11. Final test metrics

- Test loss: **0.1786**
- Test perplexity: **1.1956**
- Test token accuracy (teacher forced): **0.9444**
- Tokens scored: 1,517,269

Held-out question answering (greedy-ish sampling, temperature 0.2):

- Questions: 12
- Mean token F1: **0.8667**
- Exact match rate: **0.1667**
- Mean numeric recall: **0.0000**
- Numerals generated that do not appear in the reference answer: 6

## 12. Sample inference

**Q:** What is the grade for student 472468?
**Reference answer:** The grade is A.
**Model answer:** The grade is A.
_(token F1 1.000, numeric recall n/a)_

**Q:** What is the weekly self study hours for student 472468?
**Reference answer:** The weekly self study hours is 21.5 hours per week.
**Model answer:** The weekly self study hours is 13.1 hours per week.
_(token F1 0.900, numeric recall 0.000)_

**Q:** What is the grade for student 410603?
**Reference answer:** The grade is A.
**Model answer:** The grade is A.
_(token F1 1.000, numeric recall n/a)_

**Q:** What is the weekly self study hours for student 410603?
**Reference answer:** The weekly self study hours is 13.1 hours per week.
**Model answer:** The weekly self study hours is 17.1 hours per week.
_(token F1 0.900, numeric recall 0.000)_

**Q:** What is the grade for student 641524?
**Reference answer:** The grade is A.
**Model answer:** The grade is B.
_(token F1 0.750, numeric recall n/a)_

**Q:** What is the weekly self study hours for student 641524?
**Reference answer:** The weekly self study hours is 26 hours per week.
**Model answer:** The weekly self study hours is 18.1 hours per week.
_(token F1 0.900, numeric recall 0.000)_

**Q:** What is the grade for student 669998?
**Reference answer:** The grade is B.
**Model answer:** The grade is A.
_(token F1 0.750, numeric recall n/a)_

**Q:** What is the weekly self study hours for student 669998?
**Reference answer:** The weekly self study hours is 18.7 hours per week.
**Model answer:** The weekly self study hours is 17.3 hours per week.
_(token F1 0.900, numeric recall 0.000)_

**Q:** What is the grade for student 854661?
**Reference answer:** The grade is B.
**Model answer:** The grade is A.
_(token F1 0.750, numeric recall n/a)_

**Q:** What is the weekly self study hours for student 854661?
**Reference answer:** The weekly self study hours is 18.4 hours per week.
**Model answer:** The weekly self study hours is 17.7 hours per week.
_(token F1 0.900, numeric recall 0.000)_

## 13. Known limitations

- The model is a few-million-parameter demonstration model trained on one project's records for a very small number of steps. It reproduces the style and vocabulary of the dataset; it does not reason, and it has no world knowledge beyond the domain reference text in the training corpus.
- Generation is stochastic and can produce numbers that are not present in the source record. Numeric recall is reported above precisely so this is visible.
- Answers about a *specific* record require that record to be included in the prompt; the model has no retrieval mechanism.
- Only fields present in the dataset are learnable. Any air quality concept missing from the columns (for example a pollutant that was never measured) cannot be answered.
- CPU training limits model size and token throughput; longer training would be needed for fluent multi-sentence answers.

## 14. Future improvements

- Continue training for more steps and a larger vocabulary once GPU time is available.
- Add retrieval: put the relevant record card in the prompt for record-specific questions.
- Expand the QA templates to cover comparisons between locations and time windows.
- Evaluate against a held-out human-written question set rather than templated pairs.
- Distil a larger model's answers into this small one for better fluency.

## 15. Artifacts

- models_dir: `C:\Users\gnaya\OneDrive\Desktop\design thinking\models\student_performance_slm`
- best_model: `C:\Users\gnaya\OneDrive\Desktop\design thinking\models\student_performance_slm\best_model.pt`
- final_model: `C:\Users\gnaya\OneDrive\Desktop\design thinking\models\student_performance_slm\final_model.pt`
- tokenizer: `C:\Users\gnaya\OneDrive\Desktop\design thinking\models\student_performance_slm\tokenizer`
- checkpoints: `C:\Users\gnaya\OneDrive\Desktop\design thinking\checkpoints`

