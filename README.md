# Student Performance Prediction — automated training pipeline

An end-to-end, reproducible pipeline that loads the two Kaggle student-performance
datasets, inspects them, preprocesses them, compares candidate models, trains the best
one, evaluates it on an untouched test split, saves every artifact needed to reproduce
predictions, and answers natural-language questions from the trained model and the data.

```bash
python src/train_model.py          # inspect -> preprocess -> sanity check -> train -> test -> save
python src/predict.py --study-hours 15 --attendance 85 --participation 6.5
python src/nl_demo.py --question "What factors influence student performance?"

run_app.bat                        # web application -> http://127.0.0.1:8000
python src/ollama_setup.py --status   # local language-layer status (optional)
```

Everything in `results/` and `training_summary.md` was produced by an actual run on this
machine (Intel i5-12450H, 12 threads, 16 GB RAM, CPU only, Python 3.13, scikit-learn 1.9.1).
No metric in this repository is estimated or hand-typed.

---

## 1. Introduction

The project predicts a student's overall academic score from study behaviour. It also
classifies the letter grade. Both models are trained on the 1M-record dataset; a second,
separate model is trained on the 1,000-record exams dataset for mathematics scores.

The primary models are **gradient-boosted decision trees** (`HistGradientBoosting`),
chosen after comparing them against ridge regression, a random forest, a multilayer
perceptron and a tabular neural alternative on the same validation split. They are
**not** language models. `src/nl_demo.py` is a transparent template/retrieval interface
that reads facts from the datasets and calls the trained models for predictions; it is
not a trained language model and does not claim to be one.

## 2. Dataset sources

| Dataset | Kaggle | Local files | Rows |
|---|---|---|---|
| Student Performance – 1 Million Records | `kayhanh/student-performance-1-million` | `hugging face/million_*.parquet` | 1,000,000 |
| Students Performance in Exams | `spscientist/students-performance-in-exams` | `hugging face/exams_*.parquet` | 1,000 |

The originals are read in place and never modified. `data/raw/README.md` and
`data/raw/dataset_manifest.json` record their paths and schemas.

## 3. Dataset description

**Dataset 1 (primary)** — 6 columns, no missing values, 0 exact duplicate rows:

| column | type | notes |
|---|---|---|
| `student_id` | int | identifier, excluded from the features |
| `weekly_self_study_hours` | float | 0–40 |
| `attendance_percentage` | float | 50–100 |
| `class_participation` | float | 0–10 |
| `total_score` | float | 9.4–100, **right-censored**: 26.8 % of rows sit at exactly 100 |
| `grade` | string | A/B/C/D/F |

Target behaviour measured in `reports/data_inspection.md`:

- `grade` agrees with the documented cut-offs (A ≥ 85, B ≥ 70, C ≥ 55, D ≥ 40) for
  **99.79 %** of rows; the 2,095 exceptions are exact boundary scores assigned either way.
  Grade is therefore a near-deterministic binning of the score and is **never** an input
  to a model.
- Correlations with `total_score`: study hours **+0.812**, attendance −0.001,
  class participation −0.000. Only study hours carry linear signal.

**Dataset 2 (secondary)** — 8 columns: 5 categorical features (gender, race/ethnicity,
parental education, lunch, test-prep course) and 3 scores (math/reading/writing). No
missing values; 0 duplicate rows.

### Compatibility assessment

The two datasets share **no columns**, have no row key, use different targets (one bounded
total score vs three exam scores) and different feature types. Concatenating them would
produce rows that are 5–6/8 missing. **They are incompatible for a joint model**, so the
pipeline trains the 1M-record dataset as the primary model and the exams dataset as a
separate secondary model.

## 4. Data preprocessing (`src/preprocess.py`)

- **Splits** are the supplied `train_/validation/test` files (700k/150k/150k and
  700/150/150). Disjointness is verified: 0 shared `student_id` values between splits.
  Identical feature rows in the exams table are expected on its 600-cell categorical grid
  and carry no target information, so no re-split is done.
- **Missing values:** none exist; the loader would drop rows with missing features/target
  and reports the count rather than imputing silently.
- **Duplicates:** removed within each split (0 exact duplicates found).
- **Types:** numeric coercion for the primary table; categorical strings stripped.
- **Outliers:** counted with a 3×IQR rule and **reported, not removed** (0 outside the
  fences). Values are clipped only to their physical bounds (hours 0–40, attendance 0–100,
  participation 0–10, score 0–100).
- **Feature engineering** (row-local, no cross-row statistics): `effort_index =
  hours × participation/10`, `study_attendance_product = hours × attendance/100`,
  `participation_per_study_hour = participation/(hours+1)`.
- **Encoding/scaling:** `StandardScaler` for the ridge and MLP candidates, raw features for
  trees. The exams model uses `OneHotEncoder(handle_unknown="ignore")`. All fitted objects
  are saved, and are fitted **on the training split only**.
- **Leakage guards:** `student_id` is excluded (ID, r ≈ 0 with the target);
  `total_score` is excluded from the grade classifier; the scaler/encoder never see
  validation or test rows.

## 5. Model architecture and justification

Structured, low-dimensional tabular data with 1M rows and 6 features is the classic case
where **gradient-boosted trees** outperform both deep networks and small transformers:
no sequence structure exists, the signal is monotone and low-order, and the dataset is far
too small in feature count to justify attention. Candidate comparison on the validation
split (all trained on this machine):

| model | train rows | val MAE | val RMSE | val R² | fit time |
|---|---|---|---|---|---|
| Ridge (linear baseline) | 700,000 | 7.044 | 8.868 | 0.6701 | 0.04 s |
| Random forest (300k subsample) | 300,000 | 6.148 | 8.278 | 0.7125 | 19.2 s |
| MLP 64-32 (200k subsample) | 200,000 | 6.132 | 8.234 | 0.7156 | 8.8 s |
| **HistGradientBoosting (selected)** | **700,000** | **6.116** | **8.221** | **0.7165** | **2.0 s** |

`HistGradientBoostingRegressor` wins on accuracy *and* is ~10× cheaper than the forest;
it handles 700k rows directly, needs no GPU and produces a 0.41 MB model.

A **tabular transformer / compact LM was rejected**: with 3 informative behaviour features
there is nothing for self-attention to model, and a language model would have to serialise
rows to text and would predict digits far less accurately (the project's earlier SLM
experiment measured numeric recall 0.0 — see `docs/SLM_PIPELINE_README.md`). An MLP was
tested and placed third.

## 6. Training methodology (`src/train_model.py`)

1. Load config (`training_config.json`), seed everything (1337), detect hardware.
2. Discover and validate datasets, build processed splits, fit preprocessors.
3. **Sanity check** on 5,000 train / 2,000 validation rows — 11 assertions: data loaded,
   features exclude the target, no missing values, preprocessor shapes, finite values,
   prediction shape, finite metrics, checkpoint save/load round-trip, MLP trains and its
   loss decreases, grade classifier predicts known labels. Any failure raises and stops
   the run.
4. Train every enabled candidate on the train split, evaluate on validation.
5. Hyper-parameter search over 3 `HistGradientBoosting` configurations (depth, learning
   rate, L2), selected on validation MAE.
6. Select the best model on **validation RMSE**; save a checkpoint per candidate.
7. **Refit** the chosen configuration on train+validation (850,000 rows), then evaluate
   once on the untouched test split.
8. Train the grade classifier (behaviour features only) on train, select on validation
   weighted-F1, refit on train+validation, evaluate on test.
9. Train the secondary exams model; permutation importance on the test split; sample
   predictions; write metrics, predictions, plots and `training_summary.md`.

Reproducibility: single seed, deterministic tree settings, resumable checkpoints of every
candidate, and `training_config.json` capturing all hyper-parameters.

## 7. Evaluation metrics and results

**Regression — final model refit on 850,000 rows (train+validation), scored on 150,000 unseen test records:**

| metric | value |
|---|---|
| MAE | **6.119 points** |
| MSE | 67.346 |
| RMSE | **8.206 points** |
| R² | **0.7175** |
| target mean ± SD | 84.29 ± 15.44 |
| predictions outside 0–100 | 0 |

Validation (150k rows, before the refit): MAE 6.118, RMSE 8.221, R² 0.7165 — the test
result is consistent, with no sign of overfitting.

**Classification — grade from study behaviour only (no score input):**

| metric | value |
|---|---|
| accuracy | **0.6956** |
| weighted F1 | 0.6869 |
| macro F1 | 0.4536 |
| macro precision / recall | 0.5059 / 0.4405 |
| ROC-AUC (OVR macro) | **0.9034** |

Confusion matrix (rows = actual F, D, C, B, A; columns = predicted):

| actual \ predicted | F | D | C | B | A |
|---|---|---|---|---|---|
| F | 7 | 611 | 303 | 9 | 0 |
| D | 19 | 2,381 | 3,629 | 702 | 19 |
| C | 11 | 1,632 | 9,585 | 8,778 | 1,291 |
| B | 0 | 154 | 4,997 | 19,734 | 13,841 |
| A | 0 | 38 | 513 | 9,113 | 72,633 |

Rare grades collapse into their neighbours — expected, because the behaviour features
barely separate F/D/C (see the R² ceiling discussion below).

**Secondary exams model (math score, 700 training rows):** MAE 10.96, RMSE 13.18, R² 0.1104
on 150 test rows; coarse category accuracy 50 %. A linear baseline beat the booster on
validation and was selected. This model is a reference only.

**Feature importance (permutation on the test split):** study hours 11.74 RMSE points;
`participation_per_study_hour` 0.007; `study_attendance_product` 0.006; `effort_index`
0.004; class participation 0.0002; attendance −0.0003.

Plots in `results/visualizations/`: actual vs predicted, residuals, feature importance,
model comparison, confusion matrix, training/validation curves, exams model.

## 8. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Or just the pipeline's needs: `pip install scikit-learn joblib matplotlib pandas pyarrow`.
This machine's `python` on `PATH` is Python 2.7 (MGLTools); use `.venv\Scripts\python.exe`.

## 9. Start training

```bash
python src/train_model.py             # full run, ~1.5 minutes on this CPU
python src/train_model.py --quick     # 20 s smoke run on subsamples
python src/train_model.py --skip-sanity
python src/train_model.py --reports-only
python src/inspect_data.py            # inspection report only
python src/preprocess.py              # rebuild processed splits only
```

Measured wall clock for the full pipeline: **1 m 21 s** (700k rows, 4 candidates, 3 search
trials, 2 grade candidates, exams model, plots).

## 10. Run predictions

```bash
# one student
python src/predict.py --study-hours 15 --attendance 85 --participation 6.5

# JSON / CSV batch
python src/predict.py --json '{"study_hours": 18, "attendance": 92, "participation": 8}'
python src/predict.py --input my_students.csv --out results/my_predictions.csv

# rows from the test split, with actual values for comparison
python src/predict.py --sample 5

# secondary exams model
python src/predict.py --exams-json '{"gender": "female", "race/ethnicity": "group C",
  "parental level of education": "some college", "lunch": "standard",
  "test preparation course": "completed"}'
```

Natural-language demo (template/retrieval layer, **not** a language model):

```bash
python src/nl_demo.py --question "What factors influence student performance?"
python src/nl_demo.py --question "How does test preparation relate to exam performance?"
python src/nl_demo.py --question "Predict the score for a student studying 18 hours with 92% attendance and participation 8"
python src/nl_demo.py --interactive
```

Every answer is tagged `OBSERVATION` (computed from the dataset), `PREDICTION` (model
estimate), or `MODEL INTERPRETATION` (measured on the model).

## 11. Where the artifacts live

```
data/
  raw/            manifest + README (originals stay in hugging face/)
  processed/tabular/   million_{train,validation,test}.parquet, exams_*.parquet
  splits/         per-split id/row manifests
models/student_performance/
  best_model/     model.joblib, grade_model.joblib, exams_model.joblib
  final_model/    model.joblib (refit on train+validation)
  checkpoints/    ridge, random_forest, hist_gradient_boosting, mlp, final_refit, grade_*
  preprocessor/   million_preprocessor.joblib, exams_preprocessor.joblib,
                  preprocessor_meta.json, feature_names.json
  model_config.json
results/
  metrics.json        all measured metrics, candidates, search, importance, samples
  predictions.csv     150,000 test rows with actual/predicted/residual/predicted grade
  run_results.json    full run record used to generate training_summary.md
  data_inspection.json, preprocessing_summary.json, evaluation_rerun.json
  visualizations/*.png
training_config.json
training_summary.md
notebooks/student_performance_training.ipynb
```

Model serialisation: `joblib` (scikit-learn's portable format). Each artifact stores the
fitted model, feature order, target, scaling flag, grade labels and its metrics, so
`src/predict.py` reproduces the training-time predictions exactly.

## 12. Limitations and future improvements

- **Target ceiling.** 26.8 % of scores are exactly 100, so the regression is right-censored
  and R² saturates near 0.72. A censored/Tobit model or a two-stage "will hit 100?" model
  would model this properly.
- **Weak covariates.** Attendance and class participation have essentially zero signal in
  this dataset (|r| < 0.001). R² ≈ 0.72 is close to the information limit of the three
  features, not a tuning failure. Real improvement needs better features.
- **Grade is derived from the score.** 99.79 % of grades follow the cut-offs, so grade
  classification from behaviour alone is intrinsically harder and confuses neighbouring
  letters. Do not use it for high-stakes decisions.
- **Synthetic-looking data.** The 1M table's smooth distributions and the deterministic
  grade rule suggest a generated dataset; conclusions may not transfer to real students.
- **Small secondary dataset.** 1,000 exam records support only a coarse reference model
  (R² 0.11) and are trained separately by necessity.
- **No causal claims.** The test-prep comparison is a descriptive group difference, not an
  effect estimate.
- Future work: quantile/censored regression, calibration of grade probabilities, SHAP
  attributions, cross-validation on the exams table, and a proper calibrated deployment API.

---

## 13. Interactive application

`app/server.py` (Python standard library only, no Flask/FastAPI) serves a single-page UI at
**http://127.0.0.1:8000**:

```bash
run_app.bat
# or: .venv\Scripts\python.exe app\server.py --open
```

It contains a student input form, the predicted score and grade with an expected range, grade
probabilities, permutation importance, model what-if estimates, a natural-language assistant,
the measured metrics, the dataset summary, the limitations, and the secondary exams model.

API: `GET /api/health`, `GET /api/meta`, `POST /api/predict`, `POST /api/ask`, `POST /api/exams`.
Invalid input returns `400` with a JSON error; the UI shows it instead of a prediction.

## 14. Language layer (Ollama)

```
predict -> HistGradientBoosting (joblib)          numbers
question -> Ollama llama3.2:3b (local)             wording
```

The trained model computes every number; Ollama only writes the explanation, and only from
tool results. If Ollama is not running the assistant falls back to the deterministic engine
and says so. See `docs/OLLAMA_INTEGRATION.md`.

```bash
# one-time setup (requires Ollama installed and running)
python src/ollama_setup.py --all        # pull llama3.2:3b, write Modelfile, create the model
python src/ollama_setup.py --status
python src/ollama_service.py --ask "What factors influence student performance?"
```

| component | file |
|---|---|
| grounded prediction/statistics engine | `src/insight.py` |
| Ollama client, tools, fallback, numeric audit | `src/ollama_service.py` |
| system prompt / template / parameters | `src/ollama_prompt.py` |
| model registration | `src/ollama_setup.py` |
| generated Modelfile | `models/ollama/Modelfile.student-performance` |

`models/ollama/student-performance-slm.f16.gguf` is the GGUF export of the earlier from-scratch
2.56 M-parameter Transformer experiment. It was converted successfully and is preserved, but it
is not used as the assistant: the trained prediction model stays in joblib format, and the
language layer is a supported Llama-family model.

## 15. Tests

```bash
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

25 tests: artifact presence, metric consistency with `model_config.json`, single/batch prediction
reproducibility, input validation, what-if math, grounded answers, the HTTP API including 400/403/404
paths, Ollama tool execution, fallback and numeric audit. The LLM test skips itself when no Ollama
model is installed.

## 16. Documentation

| document | contents |
|---|---|
| `docs/ARCHITECTURE.md` | component map, data flow, leakage guards, model identification |
| `docs/DATASETS.md` | both datasets, quality checks, compatibility |
| `docs/TRAINING.md` | pipeline steps, candidates, hyper-parameters, reproducibility |
| `docs/EVALUATION.md` | measured metrics, confusion matrix, importance, cross-checks |
| `docs/OLLAMA_INTEGRATION.md` | tool contract, grounding safeguards, setup |
| `docs/APPLICATION.md` | UI screens, API, tests |
| `docs/LIMITATIONS.md` | measured limits and future work |
| `docs/SLM_PIPELINE_README.md` | the original from-scratch SLM experiment |

---

## Project layout

```
app/
  server.py       HTTP server + JSON API (stdlib only)
  static/         index.html, styles.css, app.js
tests/
  test_system.py  end-to-end suite
src/
  train_model.py    single entry point for the whole pipeline
  inspect_data.py   dataset + environment inspection report
  preprocess.py     loading, cleaning, feature engineering, encoding, splits
  evaluate.py       metrics, artifacts export, plots (re-runnable)
  predict.py        inference CLI / Python API
  insight.py        grounded predictions, statistics, what-if, deterministic answers
  ollama_service.py Ollama tool-calling bridge (numbers come from the joblib model)
  ollama_setup.py   pull base model / write Modelfile / register the Ollama model
  ollama_prompt.py  canonical system prompt and parameters
  nl_demo.py        natural-language demo over data + trained models
  utils.py          shared helpers (seeding, hardware, metrics, dataset discovery)
  train_slm.py ...  the earlier from-scratch SLM experiment (see docs/SLM_PIPELINE_README.md)
notebooks/student_performance_training.ipynb
```

The earlier decoder-only SLM experiment is preserved and documented in
`docs/SLM_PIPELINE_README.md`; it is independent of this tabular pipeline.
