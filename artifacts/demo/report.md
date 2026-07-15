# Demo training report

Generated at `2026-07-15T21:22:33.699000+00:00` using `synthetic_demo` data.

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

## On-time model without weather

- Brier score: `0.2440` (lower is better)
- Naive-rate baseline Brier score: `0.2428`
- ROC AUC: `0.5483`
- Log loss: `0.6817`

The weather-enhanced model is used only for usable live or forecast weather.
All other weather states select this separate no-weather model; no proxy value is
inserted into the prediction.

> These numbers describe a deterministic synthetic-data demo. They are pipeline checks,
> not evidence of production performance. Retrain and re-evaluate on representative data.
