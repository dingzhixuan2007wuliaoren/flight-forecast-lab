# Demo training report

Generated at `2026-09-03T01:50:16.504165+00:00` using `synthetic_demo` data.

## Fare model

- Test MAE: `$73.04`
- Naive median baseline MAE: `$365.88`
- RMSE: `$119.89`
- R²: `0.955`
- 80% conformal interval half-width: `$105.99`
- Empirical interval coverage: `0.798`
- Factor set selected on the selection window: `legacy`
- Selected factors: `none`
- Legacy selection-window MAE: `$74.18`
- Enriched selection-window MAE: `$73.78`
- Enriched selection-window paired MAE improvement: `$0.40` (`98.3%` Bonferroni-adjusted block-bootstrap interval `$-1.76` to `$2.65`)
- Held-out selected-vs-legacy improvement: not applicable (legacy retained; zero by definition)

## On-time model

- Brier score: `0.2347` (lower is better)
- Naive-rate baseline Brier score: `0.2429`
- ROC AUC: `0.6131`
- Log loss: `0.6625`

## On-time model without weather

- Brier score: `0.2433` (lower is better)
- Naive-rate baseline Brier score: `0.2429`
- ROC AUC: `0.5423`
- Log loss: `0.6801`

The weather-enhanced model is used only for usable live or forecast weather.
All other weather states select this separate no-weather model; no proxy value is
inserted into the prediction.

> These numbers describe a deterministic synthetic-data demo. They are pipeline checks,
> not evidence of production performance. Retrain and re-evaluate on representative data.
