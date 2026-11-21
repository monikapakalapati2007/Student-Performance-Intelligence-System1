# Limitations and future improvements

## Model limitations (measured)

1. **Only three behaviour features.** The model knows weekly self-study hours, attendance
   percentage and class participation. It knows nothing about subject, teacher, prior
   attainment, health, home circumstances or assessment type.
2. **Attendance and participation have no measured signal.** |r| < 0.001 with the target and
   permutation importance ≈ 0. Any claim that raising attendance raises marks is *not*
   supported by this dataset or this model.
3. **Right-censored target.** 26.8 % of scores are exactly 100. The regressor cannot separate a
   student at the ceiling from others, which caps R² near 0.72.
4. **Grade is derived from the score.** 99.79 % of grades follow the cut-offs, so a grade model
   that cannot see the score is intrinsically weaker (accuracy 0.696, macro F1 0.454) and
   confuses neighbouring letters. Do not use it for high-stakes decisions.
5. **Secondary model is coarse.** 1,000 exam rows over a 600-cell grid give R² 0.11. It is a
   reference, not a predictor.
6. **Synthetic-looking data.** The 1M table's smooth distributions and deterministic grade rule
   suggest generated data. Conclusions may not transfer to real students.
7. **Associations, not causes.** Every statement in the UI is a statistical association or a
   model what-if estimate, not a causal effect.

## System limitations

8. **The LLM can still mis-word a fact.** Grounding is enforced by prompt, tool-only numbers,
   the always-visible deterministic answer and a numeric audit — but a small 1B model may
   paraphrase awkwardly. The numbers in the evidence panel are authoritative.
9. **The original SLM is not an assistant.** A 2.56 M-parameter from-scratch Transformer cannot
   answer free-form questions; its GGUF export is preserved for the report, not for use.
10. **No authentication or multi-user persistence.** The app is a local demonstration server on
    loopback. Do not expose it to a network.
11. **CPU-only in this environment.** The RTX 3050 was detected but the installed PyTorch build
    is CPU-only (`2.15.0+cpu`), so both the tabular models and the SLM were trained on CPU.
    This is sufficient: the primary model trains in ~2 s.

## Future improvements

- Censored (Tobit) or two-stage "will reach 100?" regression for the ceiling.
- Calibrated grade probabilities and cost-sensitive thresholds.
- SHAP attributions per student for richer explanations.
- Cross-validation on the exams table and additional exam-side features.
- Optional class-level analytics (cohort dashboards) once real, permitted student data exists.
- Package the deterministic engine as a small REST service with authentication before any
  deployment beyond a local demo.
