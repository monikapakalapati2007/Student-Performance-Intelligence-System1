# Training

Single entry point:

```bash
.venv\Scripts\python.exe src\train_model.py       # full run, ~1 min 22 s on this machine
```

## Steps performed by `src/train_model.py`

1. **Config merge** — `training_config.json` merged with CLI overrides; every seed set to 1337
   (`utils.set_seed`); hardware detected (CPU/GPU, RAM, VRAM) and recorded in `results/metrics.json`.
2. **Dataset discovery** — `utils.discover_datasets` finds tables by *schema signature*, not by
   filename, so the originals in `hugging face/` are picked up without copying them.
3. **Preprocessing** — `preprocess.run_preprocessing` cleans, engineers three row-local
   features, fits the scaler/encoder on the training split only, verifies split disjointness,
   and writes processed parquet splits.
4. **Sanity check** (11 assertions on 5,000 train / 2,000 validation rows) — data loaded,
   target excluded from features, no missing values, preprocessor shape, finite values,
   prediction shape, finite metrics, checkpoint save/load round-trip, MLP trains with decreasing
   loss, grade classifier predicts known labels. A failure aborts before full training.
5. **Candidate training** on the 700k training split:

   | candidate | train rows | val MAE | val RMSE | val R² |
   |---|---|---|---|---|
   | ridge | 700,000 | 7.044 | 8.868 | 0.6701 |
   | random forest (300 trees) | 300,000 | 6.148 | 8.278 | 0.7125 |
   | MLP 64-32 | 200,000 | 6.132 | 8.234 | 0.7156 |
   | **HistGradientBoosting** | 700,000 | **6.116** | **8.221** | **0.7165** |

6. **Hyper-parameter search** — three `HistGradientBoosting` configurations
   (`max_iter` 300–600, `learning_rate` 0.03–0.06, `max_leaf_nodes` 15–31, `l2` 0.5–2.0),
   selected on validation MAE. Winner: `max_iter=500`, `learning_rate=0.04`,
   `max_leaf_nodes=15`, `min_samples_leaf=80`, `l2_regularization=2.0`.
7. **Selection** on validation RMSE, checkpoint saved per candidate in
   `models/student_performance/checkpoints/`.
8. **Refit** the winner on train + validation (850,000 rows), then a single evaluation on the
   untouched 150,000-row test split (no test-based selection at any point).
9. **Grade classifier** — HistGradientBoosting on the same behaviour features (never the
   score); selected on validation weighted F1, refit on train+validation, evaluated on test.
10. **Secondary exams model** — linear regression vs HistGradientBoosting on the 700-row exams
    training split; the linear model won on validation, so it was kept.
11. **Permutation importance** on the test split, sample predictions, plots, `results/metrics.json`,
    `results/predictions.csv`, `results/run_results.json`, `training_summary.md`.

## Flags

| flag | effect |
|---|---|
| `--quick` | 20 s smoke run on subsamples (does not overwrite final artifacts) |
| `--skip-sanity` | skip the sanity gate (not recommended) |
| `--reports-only` | regenerate reports/plots from existing artifacts |
| `--max-train-rows N` | cap the training rows |

## Reproducibility

- Fixed seeds (`1337`) for NumPy, Python and scikit-learn.
- Deterministic tree fitting (`early_stopping` uses an internal random split with the fixed seed).
- All preprocessing objects saved as joblib next to the models, so inference uses the exact
  training-time scaler, encoder and feature order.
- Model artifacts: `models/student_performance/{best_model,final_model,checkpoints,preprocessor}`.
  The primary model is 0.41 MB.

## Verification status

The pipeline was executed end to end on this machine (CPU only, scikit-learn 1.9.1) three
times, including once after the final code changes. `results/evaluation_rerun.json` records a
re-run of `src/evaluate.py` that reproduced the same test metrics from the saved artifacts.
