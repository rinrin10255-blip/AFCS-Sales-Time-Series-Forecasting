# AFCS Sales Forecasting (FOODS_3 / TX_3)

School project for item-level sales forecasting (28-day horizon) on the M5 subset.
The code focuses on **FOODS_3** items in **TX_3** store.

## Data files (in `data/`)
- `calendar_afcs2025.csv`
- `sell_prices_afcs2025.csv`
- `sales_train_validation_afcs2025.csv`
- `sales_test_validation_afcs2025.csv`
- `sales_test_evaluation_afcs_2025.csv` (evaluation only, do not train on this)

## Quick run

### 1) EDA plots
```bash
python EDA/eda_afcs2025.py
```
Outputs go to `eda_outputs/` (PNG files).

### 2) Forecasting pipeline
```bash
python forecasting/forecasting_afcs2025.py
```
Outputs go to `forecasts/`, including:
- `model_comparison_validation.csv`
- `validation_total_actual_vs_pred_<MODEL>.png`
- `test_predictions_item_level.csv`
- `sarimax_summary.txt`, `sarimax_params.csv`, `sarimax_ljungbox.csv`

### 3) Report figures (best model)
```bash
python forecasting/report_figures_ets.py
```
Optional sample item:
```bash
python forecasting/report_figures_ets.py FOODS_3_001_TX_3_validation
```
Outputs in `forecasts/`:
- `model_comparison_validation.png`
- `validation_forecast_sample_item_<MODEL>.png`
- `residuals_sample_item_<MODEL>.png`

## Notes
- The best model is selected from the validation comparison table.
- SARIMAX may show convergence warnings for some items; this is common with noisy series.
- The evaluation file is **only** for final testing, never for training.

## Requirements
Python libraries:
`pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`.
