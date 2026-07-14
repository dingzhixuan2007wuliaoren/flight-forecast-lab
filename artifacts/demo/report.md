# Demo training report

Generated at `2026-07-14T20:57:17.073044+00:00` using `synthetic_demo` data.

## Fare model

- Test MAE: `$73.99`
- Naive median baseline MAE: `$366.63`
- RMSE: `$119.54`
- R²: `0.955`
- 80% conformal interval half-width: `$110.18`
- Empirical interval coverage: `0.812`

## On-time model

- Brier score: `0.2345` (lower is better)
- Naive-rate baseline Brier score: `0.2428`
- ROC AUC: `0.6132`
- Log loss: `0.6622`

> These numbers describe a deterministic synthetic-data demo. They are pipeline checks,
> not evidence of production performance. Retrain and re-evaluate on representative data.
