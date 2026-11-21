# Datasets

Both datasets are Kaggle downloads stored in `hugging face/`. They are read in place and
**never modified, moved or regenerated**. `data/raw/` contains only a README and a manifest
with paths, checksums and schemas.

## Dataset 1 — Student Performance (1 Million Records)

Kaggle: `kayhanh/student-performance-1-million`
Files: `hugging face/million_train.parquet` (700,000), `million_validation.parquet`
(150,000), `million_test.parquet` (150,000).

| column | type | range / values | role |
|---|---|---|---|
| `student_id` | int64 | unique | identifier — excluded from features |
| `weekly_self_study_hours` | float64 | 0 – 40 | feature |
| `attendance_percentage` | float64 | 50 – 100 | feature |
| `class_participation` | float64 | 0 – 10 | feature |
| `total_score` | float64 | 9.4 – 100 | **regression target** |
| `grade` | string | F, D, C, B, A | **classification target** |

Measured properties (see `reports/data_inspection.md`):

- 1,000,000 rows, 0 missing cells, 0 exact duplicate rows.
- `total_score` is **right-censored**: 26.8 % of rows equal exactly 100.
- `grade` matches the documented cut-offs A ≥ 85, B ≥ 70, C ≥ 55, D ≥ 40 for **99.79 %** of
  rows; the remaining 2,095 are exact boundary scores labelled either way. Grade is therefore
  a near-deterministic binning of the score — it is never used as a model input.
- Pearson correlation with `total_score`: study hours **+0.812**, attendance −0.001,
  class participation −0.000.
- Splits are disjoint: 0 shared `student_id` values between train/validation/test.

Consequence: the achievable R² is bounded by the information in study hours plus a censored
target. 0.7175 measured on 150,000 held-out rows is close to that bound, not a tuning failure.

## Dataset 2 — Students Performance in Exams

Kaggle: `spscientist/students-performance-in-exams`
Files: `hugging face/exams_train.parquet` (700), `exams_validation.parquet` (150),
`exams_test.parquet` (150).

| column | type | values |
|---|---|---|
| `gender` | category | female / male |
| `race/ethnicity` | category | group A – group E |
| `parental level of education` | category | 6 levels |
| `lunch` | category | standard / free-reduced |
| `test preparation course` | category | completed / none |
| `math score`, `reading score`, `writing score` | int | 0 – 100 |

1,000 rows, 0 missing values, 0 duplicates. Feature grid is 5 × 5 × 6 × 2 × 2 = 600 cells, so
identical feature rows necessarily appear across splits; there is no student key and no target
information is shared, so the provided splits are kept.

## Compatibility

The two datasets share **no columns, no key and no target** and use different feature types.
A joint table would be 5–6 of 8 columns empty per row. They are therefore modelled separately:

- primary: 1M dataset → `total_score` regression + `grade` classification;
- secondary: exams dataset → `math score` regression (reference only, R² 0.11).

## Processed outputs

`data/processed/tabular/` holds the cleaned splits used by training:
`million_{train,validation,test}.parquet` and `exams_{train,validation,test}.parquet`.
`data/splits/` holds per-split id/row manifests for reproducibility. Nothing here is a copy of
the originals; these are deterministic transformations that can be rebuilt at any time with
`python src/preprocess.py`.
