# Holdout evaluation — FEv4 (identity-level split)

Test set: 29 images from 26 identities

| Metric | Value |
|---|---|
| Exact accuracy | **72.41%** (95% CI 54.3–85.3%) |
| Macro-F1 | 0.738 |
| Within-one-stage accuracy | 93.10% |
| Ordinal MAE (stages) | 0.345 |
| Best epoch (val loss) | 5 |

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| group1 | 0.333 | 0.600 | 0.429 | 5 |
| group2 | 0.750 | 0.600 | 0.667 | 10 |
| group3 | 1.000 | 0.750 | 0.857 | 8 |
| group4 | 1.000 | 1.000 | 1.000 | 6 |

## Confusion matrix (rows = true, cols = predicted)

| true \\ pred | group1 | group2 | group3 | group4 |
|---|---|---|---|---|
| **group1** | 3 | 2 | 0 | 0 |
| **group2** | 4 | 6 | 0 | 0 |
| **group3** | 2 | 0 | 6 | 0 |
| **group4** | 0 | 0 | 0 | 6 |

## Error distances

- Exact: 21/29
- |Δ| = 1 stage(s): 6 errors (75.0% of errors)
- |Δ| = 2 stage(s): 2 errors (25.0% of errors)
