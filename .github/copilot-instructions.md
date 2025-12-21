# Copilot instructions for AFCS Sales Forecasting

Quick, focused guidance to help an AI coding agent be productive in this repository.

1. Big picture
- Purpose: time-series forecasting for AFCS 2025 sales using item-level historical sales, calendar events and weekly prices.
- Major components:
  - `EDA/eda_afcs2025.py`: data loading, cleaning, melt/wide↔long utilities, and many plotting/diagnostic helpers used by analysis and pipeline.
  - `forecasting/forecasting_afcs2025.py`: main forecasting pipeline, `Config` (CFG) controls behavior, item-level model evaluation and final prediction export.
  - `data/`: canonical CSV inputs (`sales_train_validation_afcs2025.csv`, `calendar_afcs2025.csv`, `sell_prices_afcs2025.csv`, plus test/validation files).
  - `eda_outputs/` and `forecasts/`: generated artifacts (plots, SARIMAX diagnostics, CSV predictions).

2. Key data flow & integration points
- `EDA.load_data(include_id=True)` is a shared entrypoint: both EDA and forecasting import it. Use it to obtain `calendar`, `sales_train_wide`, `sell_prices`.
- Sales are represented in wide format in `sales_*_*.csv` with `d_1..d_N` columns; EDA provides `melt_sales_train()` to create long format and `merge_all()` to join calendar and prices.
- `forecasting` relies on calendar preprocessing utilities: `prepare_calendar()`, `add_has_event()`, `add_price_features()` and exogenous builders `exog_from_calendar_by_d()` / `daily_exog_from_calendar()`.
- Day-col mapping: functions `sorted_day_cols()` and `d_cols_to_dates()` are used repeatedly — preserve their semantics when adding features or changing date mapping.

3. Running the project (environment / commands)
- Recommended environment: conda/python 3.10+ with `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels` (>=0.14), `scipy`.
- Quick commands (run from project root):
  - Run full EDA: `python EDA/eda_afcs2025.py`
  - Run forecasting pipeline: `python forecasting/forecasting_afcs2025.py`
  - Inspect outputs: `ls forecasts/` and `ls eda_outputs/`

4. Project-specific conventions
- Use CSVs in `data/` as the canonical source; code expects specific column names: `id`, `d_*` day columns, `wm_yr_wk`, `date`, `event_name_1/2`, `snap_*`, `sell_price`.
- Config-driven pipeline: change `CFG` in `forecasting/forecasting_afcs2025.py` to alter `focus_dept_id`, `focus_store_id`, `models`, `order`, `seasonal_order`, and output dirs.
- Treat `d_*` columns as authoritative for mapping to calendar `d` values; never infer dates by position unless using `d_cols_to_dates()`.
- Error-handling pattern: forecasting falls back to `SeasonalNaive` when model fitting fails (see `evaluate_models_item_level`) — mirror this safe default when adding new models.

5. Typical edits an agent may perform
- Add an exogenous feature: update `CFG.exog_cols`, implement a calendar transform (e.g., lagged price) in `forecasting.add_price_features()` and update `exog_from_calendar_by_d()` usage.
- Add a new model: implement `fit_<model>_and_forecast(...)`, register name in `predict_model()` and include it in `CFG.models`.
- Speed-ups for local dev: reduce `CFG.max_items` and `CFG.progress_every` to run smaller samples quickly.

6. Files to inspect for examples
- `EDA/eda_afcs2025.py` — data parsing, `melt_sales_train`, `merge_all`, plotting helpers.
- `forecasting/forecasting_afcs2025.py` — `Config` usage, exogenous builders, model training, diagnostics export (`save_sarimax_diagnostics`).
- `forecasts/` — examples of saved diagnostics: `sarimax_ljungbox.csv`, `sarimax_params.csv`, `sarimax_summary.txt`.

7. What to avoid / gotchas
- Do not assume `calendar` has a 'd' column; use `prepare_calendar()` to canonicalize dates.
- Many functions expect numeric `exog` frames with matching indices — use `prepare_exog()` and ensure `reindex` uses the same date index.
- Large runs are slow and may fail silently on numeric/optimization warnings; keep `warnings` filters in mind and prefer test runs on a subset.

If anything above is unclear or you'd like me to expand examples (small code snippets or tests), say which area to iterate on.
