# Raw data (read-only)

The original Kaggle downloads are **not copied or modified**. The pipeline reads them
in place from `hugging face/` (the folder where they were downloaded), so these files
stay exactly as supplied:

| File | Rows | Role |
|---|---|---|
| `hugging face/million_train_.parquet` | 700,000 | dataset 1 — train |
| `hugging face/million_validation.parquet` | 150,000 | dataset 1 — validation |
| `hugging face/million_test.parquet` | 150,000 | dataset 1 — test |
| `hugging face/exams_train.parquet` | 700 | dataset 2 — train |
| `hugging face/exams_validation.parquet` | 150 | dataset 2 — validation |
| `hugging face/exams_test.parquet` | 150 | dataset 2 — test |

Sources:
- https://www.kaggle.com/datasets/kayhanh/student-performance-1-million
- https://www.kaggle.com/datasets/spscientist/students-performance-in-exams

`dataset_manifest.json` records the exact paths, row counts and column schemas
produced by `python src/inspect_data.py`. Processed, model-ready copies are written to
`data/processed/tabular/`, leaving the originals untouched.
