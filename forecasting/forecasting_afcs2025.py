# %%
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import NormalDist
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from EDA.eda_afcs2025 import load_data, prepare_calendar, melt_sales_train, merge_all 

@dataclass
class Config:
    # paths
    data_dir: str = "data"
    out_dir: str = "forecasts"
    
    project_root: str = PROJECT_ROOT

    # files (in /data)
    train_file: str = "sales_train_validation_afcs2025.csv"
    test_valid_file: str = "sales_test_validation_afcs2025.csv"
    test_eval_file: str = "sales_test_evaluation_afcs_2025.csv" 

    horizon: int = 28
    seasonal_period: int = 7

    # SARIMAX orders
    order: Tuple[int, int, int] = (1, 0, 1)
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7)

    # exogenous variables from calendar
    exog_cols: Tuple[str, ...] = (
        "wday",
        "month",
        "year",
        "snap_TX",
        "has_event",
        "avg_sell_price",
    )
    save_diagnostics: bool = True
    save_selection_plots: bool = True

CFG = Config()

# Utilities
def ensure_dirs() -> str:
    out_path = os.path.join(PROJECT_ROOT, CFG.out_dir)
    os.makedirs(out_path, exist_ok=True)
    return out_path

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def add_has_event(calendar_df: pd.DataFrame) -> pd.DataFrame:
    c = calendar_df.copy()

    e1 = "event_name_1" in c.columns
    e2 = "event_name_2" in c.columns

    if e1 and e2:
        c["has_event"] = (c["event_name_1"].notna()) | (c["event_name_2"].notna())
    elif e1:
        c["has_event"] = c["event_name_1"].notna()
    elif e2:
        c["has_event"] = c["event_name_2"].notna()
    else:
        c["has_event"] = 0

    c["has_event"] = c["has_event"].astype(int)
    return c

def add_price_features(calendar_df: pd.DataFrame, sell_prices: pd.DataFrame) -> pd.DataFrame:
    c = calendar_df.copy()
    if "wm_yr_wk" not in c.columns or "wm_yr_wk" not in sell_prices.columns:
        return c

    weekly_price = (
        sell_prices
        .groupby("wm_yr_wk", as_index=False)["sell_price"]
        .mean()
        .rename(columns={"sell_price": "avg_sell_price"})
    )
    c = c.merge(weekly_price, on="wm_yr_wk", how="left")
    return c

def aggregate_total_daily_sales_from_merged(df_merged: pd.DataFrame) -> pd.Series:
    """Merged long DF -> total sales per date."""
    daily = (
        df_merged
        .groupby("date", as_index=True)["sales"]
        .sum()
        .sort_index()
    )
    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq("D")
    daily = daily.fillna(0)

    return daily

def sorted_day_cols(df: pd.DataFrame) -> list[str]:
    d_cols = [c for c in df.columns if str(c).startswith("d_")]
    return sorted(d_cols, key=lambda x: int(str(x).split("_")[1]))

def prepare_exog(exog: pd.DataFrame) -> pd.DataFrame:
    exog = exog.ffill().bfill().fillna(0)
    for c in exog.columns:
        exog[c] = pd.to_numeric(exog[c], errors="coerce").fillna(0)
    return exog

def daily_exog_from_calendar(calendar_prepped: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:

    cols = ["date"] + [c for c in CFG.exog_cols if c in calendar_prepped.columns]
    exog = (
        calendar_prepped[cols]
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")
    )
    exog.index = pd.to_datetime(exog.index)
    exog = exog.reindex(idx)
    exog = prepare_exog(exog)
    return exog
        
def wide_file_to_daily_total(
    sales_wide: pd.DataFrame,
    calendar_prepped: pd.DataFrame
) -> pd.Series:
    """Convert wide sales DF to total daily sales Series."""
    d_cols = sorted_day_cols(sales_wide)
    if len(d_cols) == 0:
        raise ValueError("No d_* columns found in the provided wide sales file.")

    totals = sales_wide[d_cols].sum(axis=0) 
    totals.index = totals.index.astype(str)

    # map d -> date
    mapper = (
        calendar_prepped[["d", "date"]]
        .drop_duplicates("d")
        .set_index("d")["date"]
        .to_dict()
    )
    dates = [mapper.get(d, None) for d in totals.index]
    if any(x is None for x in dates):
        missing = [totals.index[i] for i, x in enumerate(dates) if x is None][:10]
        raise ValueError(f"Some d values cannot be mapped to dates. Examples: {missing}")

    s = pd.Series(totals.values, index=pd.to_datetime(dates)).sort_index()
    s.name = "total_sales"
    return s


# Models
def seasonal_naive_forecast(y_train: pd.Series, h: int, season: int) -> np.ndarray:
   
    if len(y_train) < season:
        raise ValueError("Training series shorter than seasonal_period.")
    last_season = y_train.iloc[-season:].values
    reps = int(np.ceil(h / season))
    return np.tile(last_season, reps)[:h]

def fit_sarimax(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
):
    y = y_train.astype("float64").copy()
    y_log = np.log1p(y)
    X_train = exog_train.astype("float64")

    model = SARIMAX(
        y_log,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=200)
    return res


def fit_sarimax_and_forecast(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    exog_future: pd.DataFrame,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    h: int
) -> np.ndarray:
    res = fit_sarimax(y_train, exog_train, order, seasonal_order)
    X_future = exog_future.astype("float64")

    fcast = res.get_forecast(steps=h, exog=X_future)

    yhat_log = fcast.predicted_mean
    yhat = np.expm1(yhat_log).values

    yhat = np.clip(yhat, 0, None)

    return yhat

def fit_sarima_no_exog_and_forecast(
    y_train: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    h: int,
) -> np.ndarray:
    y = y_train.astype("float64").copy()
    y_log = np.log1p(y)

    model = SARIMAX(
        y_log,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=200)
    fcast = res.get_forecast(steps=h)

    yhat_log = fcast.predicted_mean
    yhat = np.expm1(yhat_log).values
    yhat = np.clip(yhat, 0, None)
    return yhat

def fit_ets_and_forecast(
    y_train: pd.Series,
    h: int,
    seasonal_period: int,
) -> np.ndarray:
    model = ExponentialSmoothing(
        y_train.astype("float64"),
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_period,
        initialization_method="estimated",
    )
    res = model.fit(optimized=True)
    yhat = res.forecast(h)
    yhat = np.asarray(yhat, dtype=float)
    yhat = np.clip(yhat, 0, None)
    return yhat

def save_sarimax_diagnostics(res, out_dir: str, prefix: str = "sarimax") -> None:
    summary_path = os.path.join(out_dir, f"{prefix}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(res.summary().as_text())

    params = pd.Series(res.params, index=res.param_names, name="coef")
    params.to_csv(os.path.join(out_dir, f"{prefix}_params.csv"), header=True)

    resid = pd.Series(res.resid, name="resid").dropna()
    lb = acorr_ljungbox(resid, lags=[7, 14, 21, 28], return_df=True)
    lb.to_csv(os.path.join(out_dir, f"{prefix}_ljungbox.csv"), index=False)

    try:
        fig = res.plot_diagnostics(figsize=(12, 8))
        fig.savefig(
            os.path.join(out_dir, f"{prefix}_diagnostics.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        return
    except Exception as exc:
        print(f"[WARN] Diagnostics plot failed ({exc}); writing fallback plot.")

    resid = pd.Series(res.resid, name="resid").dropna()
    if resid.empty:
        return

    mu = float(resid.mean())
    sigma = float(resid.std(ddof=1)) or 1.0
    n = len(resid)
    probs = (np.arange(1, n + 1) - 0.5) / n
    normal = NormalDist(mu, sigma)
    theo = np.array([normal.inv_cdf(p) for p in probs])
    samp = np.sort(resid.values)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(resid.index, resid.values, color="tab:blue")
    axes[0, 0].set_title("Residuals over time")

    axes[0, 1].hist(resid.values, bins=40, density=True, color="tab:gray", alpha=0.7)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    axes[0, 1].plot(x, pdf, color="tab:red")
    axes[0, 1].set_title("Residual histogram")

    axes[1, 0].scatter(theo, samp, s=10, alpha=0.6)
    axes[1, 0].plot([theo.min(), theo.max()], [theo.min(), theo.max()], color="tab:red")
    axes[1, 0].set_title("QQ plot (normal)")

    plot_acf(resid, lags=30, ax=axes[1, 1])
    axes[1, 1].set_title("Residual ACF")

    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, f"{prefix}_diagnostics.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

def save_prediction_plot(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str,
    filename: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, y_true, label="Actual", linewidth=2)
    ax.plot(dates, y_pred, label="Forecast", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close(fig)

def save_acf_pacf_plot(
    y_series: pd.Series,
    out_dir: str,
    filename: str = "acf_pacf.png",
    nlags: int = 30,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(y_series, lags=nlags, ax=axes[0])
    plot_pacf(y_series, lags=nlags, ax=axes[1])
    axes[0].set_title("ACF of total daily sales")
    axes[1].set_title("PACF of total daily sales")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close(fig)

def load_future_series_from_file(
    calendar: pd.DataFrame,
    file_name: str,
) -> pd.Series:
    path = os.path.join(PROJECT_ROOT, CFG.data_dir, file_name)
    df = pd.read_csv(path)
    return wide_file_to_daily_total(df, calendar)

def forecast_on_wide_file(
    y_train_all: pd.Series,
    calendar: pd.DataFrame,
    file_name: str,
) -> Tuple[pd.Series, np.ndarray]:
    test_path = os.path.join(PROJECT_ROOT, CFG.data_dir, file_name)
    test_df = pd.read_csv(test_path)

    y_future = wide_file_to_daily_total(test_df, calendar)

    exog_all = daily_exog_from_calendar(
        calendar, y_train_all.index.append(y_future.index)
    )
    exog_train = exog_all.loc[y_train_all.index]
    exog_future = exog_all.loc[y_future.index]

    yhat_future = fit_sarimax_and_forecast(
        y_train=y_train_all,
        exog_train=exog_train,
        exog_future=exog_future,
        order=CFG.order,
        seasonal_order=CFG.seasonal_order,
        h=len(y_future),
    )

    return y_future, yhat_future


# %%
# Main pipeline
def main():
    out_dir = ensure_dirs()
    print(f"[INFO] Project ROOT: {CFG.project_root}")
    print(f"[INFO] Outputs will be saved to: {out_dir}")

    # 1 Load base data via EDA utilities
    calendar, sales_train_wide, sell_prices = load_data()
    calendar = prepare_calendar(calendar)
    calendar = add_has_event(calendar)
    calendar = add_price_features(calendar, sell_prices)
    calendar["date"] = pd.to_datetime(calendar["date"])

    # 2 Build TRAIN merged df
    sales_long = melt_sales_train(sales_train_wide) 
    df_train_merged = merge_all(calendar, sales_long, sell_prices)
    df_train_merged["date"] = pd.to_datetime(df_train_merged["date"])

    y_all = aggregate_total_daily_sales_from_merged(df_train_merged)
    print(f"[INFO] Train daily series: {y_all.index.min().date()} -> {y_all.index.max().date()}, n={len(y_all)}")

    if CFG.save_selection_plots:
        save_acf_pacf_plot(y_all, out_dir)
        print("[INFO] Saved ACF/PACF plot.")

    exog_train = daily_exog_from_calendar(calendar, y_all.index)
    if CFG.save_diagnostics:
        res_full = fit_sarimax(
            y_train=y_all,
            exog_train=exog_train,
            order=CFG.order,
            seasonal_order=CFG.seasonal_order,
        )
        save_sarimax_diagnostics(res_full, out_dir)
        print("[INFO] Saved SARIMAX diagnostics artifacts.")

    # 3 Validation on official test_validation file (aggregate)
    y_valid = load_future_series_from_file(calendar, CFG.test_valid_file)
    exog_valid = daily_exog_from_calendar(calendar, y_valid.index)
    h = len(y_valid)

    preds = {}
    errors = {}

    def try_model(name: str, fn) -> None:
        try:
            preds[name] = fn()
        except Exception as exc:
            errors[name] = str(exc)
            print(f"[WARN] {name} failed: {exc}")

    try_model(
        "SeasonalNaive",
        lambda: seasonal_naive_forecast(y_all, h=h, season=CFG.seasonal_period),
    )
    try_model(
        "SARIMAX_exog",
        lambda: fit_sarimax_and_forecast(
            y_train=y_all,
            exog_train=exog_train,
            exog_future=exog_valid,
            order=CFG.order,
            seasonal_order=CFG.seasonal_order,
            h=h,
        ),
    )
    try_model(
        "SARIMA_no_exog",
        lambda: fit_sarima_no_exog_and_forecast(
            y_train=y_all,
            order=CFG.order,
            seasonal_order=CFG.seasonal_order,
            h=h,
        ),
    )
    try_model(
        "ETS_add",
        lambda: fit_ets_and_forecast(
            y_train=y_all,
            h=h,
            seasonal_period=CFG.seasonal_period,
        ),
    )

    results = []
    for name, pred in preds.items():
        results.append({"model": name, "rmse": rmse(y_valid.values, pred)})

    results_df = pd.DataFrame(results).sort_values("rmse")
    comparison_path = os.path.join(out_dir, "model_comparison_validation.csv")
    results_df.to_csv(comparison_path, index=False)
    print("[VAL] Model comparison (RMSE):")
    for row in results_df.itertuples(index=False):
        print(f"  {row.model}: {row.rmse:.4f}")
    print(f"[INFO] Saved model comparison -> {comparison_path}")

    best_model = results_df.iloc[0]["model"]
    best_pred = preds[best_model]

    save_prediction_plot(
        dates=y_valid.index,
        y_true=y_valid.values,
        y_pred=best_pred,
        out_dir=out_dir,
        filename=f"validation_actual_vs_pred_{best_model}.png",
        title=f"Validation: Actual vs Forecast ({best_model})",
    )
    print("[INFO] Saved validation plot.")

    # 4 Final evaluation on test_evaluation file (aggregate)
    y_test = load_future_series_from_file(calendar, CFG.test_eval_file)
    exog_test = daily_exog_from_calendar(calendar, y_test.index)

    if best_model == "SeasonalNaive":
        test_pred = seasonal_naive_forecast(
            y_all, h=len(y_test), season=CFG.seasonal_period
        )
    elif best_model == "SARIMAX_exog":
        test_pred = fit_sarimax_and_forecast(
            y_train=y_all,
            exog_train=exog_train,
            exog_future=exog_test,
            order=CFG.order,
            seasonal_order=CFG.seasonal_order,
            h=len(y_test),
        )
    elif best_model == "SARIMA_no_exog":
        test_pred = fit_sarima_no_exog_and_forecast(
            y_train=y_all,
            order=CFG.order,
            seasonal_order=CFG.seasonal_order,
            h=len(y_test),
        )
    elif best_model == "ETS_add":
        test_pred = fit_ets_and_forecast(
            y_train=y_all,
            h=len(y_test),
            seasonal_period=CFG.seasonal_period,
        )
    else:
        raise ValueError(f"Unknown best model: {best_model}")

    test_rmse = rmse(y_test.values, test_pred)
    print(f"[TEST] {best_model} RMSE on test evaluation dataset: {test_rmse:.4f}")

    # 5 No submission file is generated for this project.


if __name__ == "__main__":
    main()

# %%
