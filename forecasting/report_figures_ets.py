"""
Generate report figures for the best model (FOODS_3 / TX_3) without running the full pipeline.
Outputs are saved to the forecasts/ folder.
"""
from __future__ import annotations

import os
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "forecasts")
FORECASTING_DIR = os.path.join(PROJECT_ROOT, "forecasting")
sys.path.insert(0, FORECASTING_DIR)
try:
    from forecasting_afcs2025 import CFG
    ORDER = CFG.order
    SEASONAL_ORDER = CFG.seasonal_order
    SEASONAL_PERIOD = CFG.seasonal_period
except Exception:
    ORDER = (1, 0, 1)
    SEASONAL_ORDER = (1, 1, 1, 7)
    SEASONAL_PERIOD = 7

TRAIN_PATH = os.path.join(DATA_DIR, "sales_train_validation_afcs2025.csv")
VALID_PATH = os.path.join(DATA_DIR, "sales_test_validation_afcs2025.csv")
CALENDAR_PATH = os.path.join(DATA_DIR, "calendar_afcs2025.csv")
SELL_PRICES_PATH = os.path.join(DATA_DIR, "sell_prices_afcs2025.csv")
MODEL_COMP_PATH = os.path.join(OUT_DIR, "model_comparison_validation.csv")

FOCUS_DEPT = "FOODS_3"
FOCUS_STORE = "TX_3"
HORIZON = 28
BEST_MODEL_FALLBACK = "SARIMAX_exog"


def ensure_out_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def sorted_day_cols(cols: list[str]) -> list[str]:
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def prepare_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    calendar = calendar.copy()
    if "date" not in calendar.columns:
        raise ValueError("calendar csv must have a 'date' column.")
    calendar["date"] = pd.to_datetime(calendar["date"])
    if "d" not in calendar.columns:
        calendar = calendar.sort_values("date").reset_index(drop=True)
        calendar["d"] = "d_" + (calendar.index + 1).astype(str)
    else:
        calendar["d"] = calendar["d"].astype(str)
    return calendar


def add_has_event(calendar: pd.DataFrame) -> pd.DataFrame:
    calendar = calendar.copy()
    e1 = calendar["event_name_1"].notna() if "event_name_1" in calendar.columns else False
    e2 = calendar["event_name_2"].notna() if "event_name_2" in calendar.columns else False
    calendar["has_event"] = (e1 | e2).astype(int)
    return calendar


def filter_focus_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns:
        return df
    prefix = f"{FOCUS_DEPT}_"
    token = f"_{FOCUS_STORE}_"
    mask = df["id"].astype(str).str.startswith(prefix) & df["id"].astype(str).str.contains(token)
    return df.loc[mask].copy()


def parse_id(sample_id: str) -> tuple[str, str]:
    parts = sample_id.split("_")
    item_id = "_".join(parts[:3])
    store_id = "_".join(parts[3:5])
    return item_id, store_id


def pick_sample_id(train_df: pd.DataFrame) -> str:
    d_cols = sorted_day_cols([c for c in train_df.columns if c.startswith("d_")])
    means = train_df[d_cols].mean(axis=1)
    idx = int(means.idxmax())
    return str(train_df.loc[idx, "id"])


def fit_ets(y_train: pd.Series) -> ExponentialSmoothing:
    model = ExponentialSmoothing(
        y_train.astype("float64"),
        trend="add",
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD,
        initialization_method="estimated",
    )
    return model.fit(optimized=True)


def fit_sarimax_model(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
) -> object:
    y_values = np.asarray(y_train, dtype=float)
    y_series = pd.Series(y_values, index=exog_train.index)
    y_log = np.log1p(y_series)
    model = SARIMAX(
        y_log,
        exog=exog_train,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=200)


def fit_sarimax(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    exog_future: pd.DataFrame,
) -> tuple[np.ndarray, object]:
    res = fit_sarimax_model(y_train, exog_train)
    fcast = res.get_forecast(steps=HORIZON, exog=exog_future)
    yhat = np.expm1(fcast.predicted_mean)
    yhat = np.asarray(yhat, dtype=float)
    yhat = np.clip(yhat, 0, None)
    return yhat, res


def fit_sarima(y_train: pd.Series) -> tuple[np.ndarray, object]:
    y_log = np.log1p(y_train.astype(float))
    model = SARIMAX(
        y_log,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=200)
    fcast = res.get_forecast(steps=HORIZON)
    yhat = np.expm1(fcast.predicted_mean)
    yhat = np.asarray(yhat, dtype=float)
    yhat = np.clip(yhat, 0, None)
    return yhat, res


def fit_sarima_model(y_train: pd.Series) -> object:
    y_log = np.log1p(y_train.astype(float))
    model = SARIMAX(
        y_log,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=200)


def seasonal_naive_forecast(y_train: pd.Series, season: int) -> np.ndarray:
    last = y_train.values[-season:]
    reps = int(np.ceil(HORIZON / season))
    return np.tile(last, reps)[:HORIZON]


def plot_model_comparison() -> None:
    if not os.path.exists(MODEL_COMP_PATH):
        print(f"[WARN] Missing {MODEL_COMP_PATH}; skipping model comparison plot.")
        return
    df = pd.read_csv(MODEL_COMP_PATH)
    if "model" not in df.columns or "rmse" not in df.columns:
        print("[WARN] model_comparison_validation.csv missing required columns.")
        return

    df = df.sort_values("rmse")
    plt.figure(figsize=(6, 4))
    plt.bar(df["model"], df["rmse"], color="tab:blue")
    plt.title("Validation RMSE by model (FOODS_3 / TX_3)")
    plt.ylabel("RMSE")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "model_comparison_validation.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def load_best_model() -> str:
    if not os.path.exists(MODEL_COMP_PATH):
        print(f"[WARN] Missing {MODEL_COMP_PATH}; using fallback {BEST_MODEL_FALLBACK}.")
        return BEST_MODEL_FALLBACK
    df = pd.read_csv(MODEL_COMP_PATH)
    if "model" not in df.columns or "rmse" not in df.columns:
        print(f"[WARN] Invalid {MODEL_COMP_PATH}; using fallback {BEST_MODEL_FALLBACK}.")
        return BEST_MODEL_FALLBACK
    best = df.sort_values("rmse").iloc[0]["model"]
    return str(best)


def build_exog(
    calendar: pd.DataFrame,
    sell_prices: pd.DataFrame,
    d_cols: list[str],
    item_id: str,
    store_id: str,
) -> pd.DataFrame:
    cal = calendar.set_index("d")
    base_cols = ["wday", "month", "year", "snap_TX", "has_event"]
    exog = pd.DataFrame(index=range(len(d_cols)))
    for col in base_cols:
        if col in cal.columns:
            exog[col] = cal.loc[d_cols, col].to_numpy()
        else:
            exog[col] = 0

    if "wm_yr_wk" in cal.columns and "sell_price" in sell_prices.columns:
        weeks = cal.loc[d_cols, "wm_yr_wk"]
        price_map = (
            sell_prices[
                (sell_prices["store_id"] == store_id)
                & (sell_prices["item_id"] == item_id)
            ]
            .set_index("wm_yr_wk")["sell_price"]
        )
        price_series = weeks.map(price_map).astype(float)
        price_series = price_series.ffill().bfill().fillna(0.0)
        exog["sell_price"] = price_series.to_numpy()
    else:
        exog["sell_price"] = 0.0

    return exog


def plot_validation_forecast(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    calendar: pd.DataFrame,
    sell_prices: pd.DataFrame,
    sample_id: str,
    model_name: str,
) -> None:
    d_train = sorted_day_cols([c for c in train_df.columns if c.startswith("d_")])
    d_valid = sorted_day_cols([c for c in valid_df.columns if c.startswith("d_")])

    train_row = train_df.loc[train_df["id"] == sample_id, d_train]
    valid_row = valid_df.loc[valid_df["id"] == sample_id, d_valid]
    if train_row.empty or valid_row.empty:
        raise ValueError(f"Sample id not found in train/valid: {sample_id}")

    y_train = train_row.iloc[0].astype(float)
    y_valid = valid_row.iloc[0].astype(float)

    if model_name == "ETS_add":
        res = fit_ets(y_train)
        y_pred = res.forecast(HORIZON)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    elif model_name == "SARIMAX_exog":
        item_id, store_id = parse_id(sample_id)
        exog_train = build_exog(calendar, sell_prices, d_train, item_id, store_id)
        exog_valid = build_exog(calendar, sell_prices, d_valid, item_id, store_id)
        y_pred, _ = fit_sarimax(y_train, exog_train, exog_valid)
    elif model_name == "SARIMA_no_exog":
        y_pred, _ = fit_sarima(y_train)
    elif model_name == "SeasonalNaive":
        y_pred = seasonal_naive_forecast(y_train, season=SEASONAL_PERIOD)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    d_to_date = dict(zip(calendar["d"].astype(str), calendar["date"]))
    dates = [d_to_date.get(d) for d in d_valid]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, y_valid.values, label="Actual", linewidth=2)
    ax.plot(dates, y_pred, label=f"{model_name} forecast", linewidth=2)
    ax.set_title(f"Validation forecast ({model_name}) — {sample_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(
        OUT_DIR,
        f"validation_forecast_sample_item_{model_name}.png",
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_residuals(
    train_df: pd.DataFrame,
    calendar: pd.DataFrame,
    sell_prices: pd.DataFrame,
    sample_id: str,
    model_name: str,
) -> None:
    d_train = sorted_day_cols([c for c in train_df.columns if c.startswith("d_")])
    train_row = train_df.loc[train_df["id"] == sample_id, d_train]
    if train_row.empty:
        raise ValueError(f"Sample id not found in train: {sample_id}")

    y_train = train_row.iloc[0].astype(float)
    if model_name == "ETS_add":
        res = fit_ets(y_train)
        fitted = res.fittedvalues
        resid = y_train.values - np.asarray(fitted, dtype=float)
    elif model_name == "SARIMAX_exog":
        item_id, store_id = parse_id(sample_id)
        exog_train = build_exog(calendar, sell_prices, d_train, item_id, store_id)
        res = fit_sarimax_model(y_train, exog_train)
        fitted_log = np.asarray(res.fittedvalues, dtype=float)
        fitted = np.expm1(fitted_log)
        resid = y_train.values - fitted
    elif model_name == "SARIMA_no_exog":
        res = fit_sarima_model(y_train)
        fitted_log = np.asarray(res.fittedvalues, dtype=float)
        fitted = np.expm1(fitted_log)
        resid = y_train.values - fitted
    elif model_name == "SeasonalNaive":
        fitted = pd.Series(y_train.values).shift(SEASONAL_PERIOD).to_numpy()
        resid = y_train.values[SEASONAL_PERIOD:] - fitted[SEASONAL_PERIOD:]
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(resid, color="tab:blue")
    axes[0, 0].set_title("Residuals over time")
    axes[0, 0].set_xlabel("Time index")
    axes[0, 0].set_ylabel("Residual")

    axes[0, 1].hist(resid, bins=40, color="tab:gray", alpha=0.7, density=True)
    axes[0, 1].set_title("Residual histogram")
    axes[0, 1].set_xlabel("Residual")

    plot_acf(resid, lags=30, ax=axes[1, 1])
    axes[1, 1].set_title("Residual ACF")

    # Simple Q-Q plot without SciPy
    resid_sorted = np.sort(resid)
    n = resid_sorted.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = np.array([NormalDist().inv_cdf(p) for p in probs])
    axes[1, 0].scatter(theo, resid_sorted, s=10, alpha=0.6)
    mu = resid.mean()
    sigma = resid.std(ddof=1) or 1.0
    line = mu + sigma * theo
    axes[1, 0].plot(theo, line, color="tab:red", linewidth=1)
    axes[1, 0].set_title("Q-Q plot")
    axes[1, 0].set_xlabel("Theoretical quantiles")
    axes[1, 0].set_ylabel("Residual quantiles")

    fig.tight_layout()
    out_path = os.path.join(
        OUT_DIR,
        f"residuals_sample_item_{model_name}.png",
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def main() -> None:
    ensure_out_dir()

    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    calendar = prepare_calendar(pd.read_csv(CALENDAR_PATH))
    calendar = add_has_event(calendar)
    sell_prices = pd.read_csv(SELL_PRICES_PATH)

    train_df = filter_focus_ids(train_df)
    valid_df = filter_focus_ids(valid_df)

    sample_id = sys.argv[1] if len(sys.argv) > 1 else pick_sample_id(train_df)
    print(f"[INFO] Sample id: {sample_id}")
    model_name = load_best_model()
    print(f"[INFO] Best model: {model_name}")

    plot_model_comparison()
    plot_validation_forecast(train_df, valid_df, calendar, sell_prices, sample_id, model_name)
    plot_residuals(train_df, calendar, sell_prices, sample_id, model_name)


if __name__ == "__main__":
    main()
