# Demo training report

Generated at `2026-07-11T18:57:20.660853+00:00` using `synthetic_demo` data.

## Fare model

- Test MAE: `$31.35`
- Naive median baseline MAE: `$112.57`
- RMSE: `$48.81`
- R²: `0.931`
- 80% conformal interval half-width: `$43.02`
- Empirical interval coverage: `0.782`

## On-time model

- Brier score: `0.2236` (lower is better)
- Naive-rate baseline Brier score: `0.2375`
- ROC AUC: `0.6287`
- Log loss: `0.6398`

> These numbers describe a deterministic synthetic-data demo. They are pipeline checks,
> not evidence of production performance. Retrain and re-evaluate on representative data.
