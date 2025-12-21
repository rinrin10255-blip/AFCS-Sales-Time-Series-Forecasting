# %%
from __future__ import annotations

import os
import sys
import warnings
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
from statsmodels.tools.sm_exceptions import ValueWarning as SMValueWarning

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from EDA.eda_afcs2025 import load_data, prepare_calendar

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
        "sell_price",
    )
    save_diagnostics: bool = True
    save_selection_plots: bool = True
    item_level: bool = True
    focus_dept_id: str | None = "FOODS_3"
    focus_store_id: str | None = "TX_3"
    compare_max_items: int | None = 100
    max_items: int | None = None
    progress_every: int = 100
    models: Tuple[str, ...] = (
        "SeasonalNaive",
        "SARIMAX_exog",
        "SARIMA_no_exog",
        "ETS_add",
    )

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

def exog_from_calendar_by_d(
    calendar_prepped: pd.DataFrame,
    d_cols: list[str],
    use_range_index: bool = True,
) -> pd.DataFrame:
    cols = [c for c in CFG.exog_cols if c in calendar_prepped.columns]
    exog = (
        calendar_prepped[["d"] + cols]
        .drop_duplicates("d")
        .set_index("d")
        .reindex(d_cols)
    )
    exog = prepare_exog(exog)
    if use_range_index:
        exog = exog.reset_index(drop=True)
    return exog

def d_cols_to_dates(
    calendar_prepped: pd.DataFrame,
    d_cols: list[str],
) -> pd.DatetimeIndex:
    mapper = (
        calendar_prepped[["d", "date"]]
        .drop_duplicates("d")
        .set_index("d")["date"]
        .to_dict()
    )
    dates = [mapper.get(d, None) for d in d_cols]
    if any(x is None for x in dates):
        missing = [d_cols[i] for i, x in enumerate(dates) if x is None][:10]
        raise ValueError(f"Some d values cannot be mapped to dates. Examples: {missing}")
    return pd.to_datetime(dates)

def prepare_item_level_matrices(
    sales_train_wide: pd.DataFrame,
    test_valid_path: str,
    test_eval_path: str,
) -> tuple[
    pd.Index,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
]:
    train_df = sales_train_wide.set_index("id")
    valid_df = pd.read_csv(test_valid_path).set_index("id")
    test_df = pd.read_csv(test_eval_path).set_index("id")

    if CFG.focus_dept_id:
        if "dept_id" in train_df.columns:
            train_df = train_df.loc[train_df["dept_id"] == CFG.focus_dept_id]
        else:
            prefix = f"{CFG.focus_dept_id}_"
            train_df = train_df.loc[train_df.index.str.startswith(prefix)]

    if CFG.focus_store_id:
        if "store_id" in train_df.columns:
            train_df = train_df.loc[train_df["store_id"] == CFG.focus_store_id]
        else:
            token = f"_{CFG.focus_store_id}_"
            train_df = train_df.loc[train_df.index.str.contains(token)]

    if CFG.focus_dept_id or CFG.focus_store_id:
        focus_ids = train_df.index
        valid_df = valid_df.loc[valid_df.index.intersection(focus_ids)]
        test_df = test_df.loc[test_df.index.intersection(focus_ids)]

    common_ids = train_df.index.intersection(valid_df.index).intersection(test_df.index)
    if len(common_ids) == 0:
        raise ValueError("No common ids across train, validation, and evaluation files.")

    train_df = train_df.loc[common_ids]
    valid_df = valid_df.loc[common_ids]
    test_df = test_df.loc[common_ids]

    train_cols = sorted_day_cols(train_df)
    valid_cols = sorted_day_cols(valid_df)
    test_cols = sorted_day_cols(test_df)

    train_matrix = train_df[train_cols].to_numpy(dtype=float)
    valid_matrix = valid_df[valid_cols].to_numpy(dtype=float)
    test_matrix = test_df[test_cols].to_numpy(dtype=float)
    item_ids = train_df["item_id"].astype(str).to_numpy()
    store_ids = train_df["store_id"].astype(str).to_numpy()

    return (
        common_ids,
        train_matrix,
        valid_matrix,
        test_matrix,
        train_cols,
        valid_cols,
        test_cols,
        item_ids,
        store_ids,
    )

def build_price_matrices(
    item_ids: np.ndarray,
    store_ids: np.ndarray,
    sell_prices: pd.DataFrame,
    calendar_prepped: pd.DataFrame,
    train_cols: list[str],
    valid_cols: list[str],
    test_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_to_week = (
        calendar_prepped[["d", "wm_yr_wk"]]
        .drop_duplicates("d")
        .set_index("d")["wm_yr_wk"]
    )

    def weeks_for(cols: list[str]) -> list[int]:
        weeks = d_to_week.reindex(cols).tolist()
        if any(pd.isna(weeks)):
            missing = [cols[i] for i, w in enumerate(weeks) if pd.isna(w)][:10]
            raise ValueError(f"Missing wm_yr_wk for d columns. Examples: {missing}")
        return weeks

    weeks_train = weeks_for(train_cols)
    weeks_valid = weeks_for(valid_cols)
    weeks_test = weeks_for(test_cols)

    prices = sell_prices.copy()
    if CFG.focus_store_id:
        prices = prices.loc[prices["store_id"] == CFG.focus_store_id]

    price_map = {}
    for (store_id, item_id), grp in prices.groupby(["store_id", "item_id"]):
        s = grp.set_index("wm_yr_wk")["sell_price"].astype(float).sort_index()
        price_map[(str(store_id), str(item_id))] = s

    n_items = len(item_ids)
    price_train = np.zeros((n_items, len(train_cols)), dtype=float)
    price_valid = np.zeros((n_items, len(valid_cols)), dtype=float)
    price_test = np.zeros((n_items, len(test_cols)), dtype=float)

    def fill_prices(series: pd.Series, weeks: list[int]) -> np.ndarray:
        vec = series.reindex(weeks)
        vec = vec.ffill().bfill().fillna(0.0)
        return vec.to_numpy(dtype=float)

    for i in range(n_items):
        key = (store_ids[i], item_ids[i])
        series = price_map.get(key)
        if series is None:
            continue
        price_train[i, :] = fill_prices(series, weeks_train)
        price_valid[i, :] = fill_prices(series, weeks_valid)
        price_test[i, :] = fill_prices(series, weeks_test)

    return price_train, price_valid, price_test
        
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

def predict_model(
    name: str,
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    exog_future: pd.DataFrame,
    h: int,
) -> np.ndarray:
    if name == "SeasonalNaive":
        return seasonal_naive_forecast(y_train, h=h, season=CFG.seasonal_period)
    if name == "SARIMAX_exog":
        return fit_sarimax_and_forecast(
            y_train=y_train,
            exog_train=exog_train,
            exog_future=exog_future,
            order=CFG.order,
            seasonal_order=CFG.seasonal_order,
            h=h,
        )
    if name == "SARIMA_no_exog":
        return fit_sarima_no_exog_and_forecast(
            y_train=y_train,
            order=CFG.order,
            seasonal_order=CFG.seasonal_order,
            h=h,
        )
    if name == "ETS_add":
        return fit_ets_and_forecast(
            y_train=y_train,
            h=h,
            seasonal_period=CFG.seasonal_period,
        )
    raise ValueError(f"Unknown model: {name}")

def evaluate_models_item_level(
    train_matrix: np.ndarray,
    valid_matrix: np.ndarray,
    base_exog_train: pd.DataFrame,
    base_exog_valid: pd.DataFrame,
    price_train: np.ndarray,
    price_valid: np.ndarray,
    models: Tuple[str, ...],
    max_items: int | None = None,
    progress_every: int = 100,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    n_items = train_matrix.shape[0]
    if max_items is not None:
        n_items = min(n_items, max_items)

    h = valid_matrix.shape[1]
    preds: dict[str, np.ndarray] = {
        name: np.zeros((n_items, h), dtype=float) for name in models
    }
    fail_counts = {name: 0 for name in models}

    for i in range(n_items):
        y_train = train_matrix[i]
        if np.all(y_train == 0):
            for name in models:
                preds[name][i, :] = 0.0
            continue

        y_train_series = pd.Series(y_train, index=base_exog_train.index)
        naive_pred = seasonal_naive_forecast(
            y_train_series, h=h, season=CFG.seasonal_period
        )

        exog_train_i = base_exog_train.copy()
        exog_valid_i = base_exog_valid.copy()
        exog_train_i["sell_price"] = price_train[i]
        exog_valid_i["sell_price"] = price_valid[i]

        for name in models:
            if name == "SeasonalNaive":
                preds[name][i, :] = naive_pred
                continue
            try:
                preds[name][i, :] = predict_model(
                    name=name,
                    y_train=y_train_series,
                    exog_train=exog_train_i,
                    exog_future=exog_valid_i,
                    h=h,
                )
            except Exception as exc:
                fail_counts[name] += 1
                preds[name][i, :] = naive_pred

        if progress_every and (i + 1) % progress_every == 0:
            print(f"[INFO] Processed {i + 1}/{n_items} items")

    y_true = valid_matrix[:n_items].reshape(-1)
    results = []
    for name in models:
        y_pred = preds[name].reshape(-1)
        results.append(
            {
                "model": name,
                "rmse": rmse(y_true, y_pred),
                "failures": fail_counts[name],
                "items_used": n_items,
            }
        )

    results_df = pd.DataFrame(results).sort_values("rmse")
    return results_df, preds

def predict_best_model_item_level(
    train_matrix: np.ndarray,
    base_exog_train: pd.DataFrame,
    base_exog_future: pd.DataFrame,
    price_train: np.ndarray,
    price_future: np.ndarray,
    model_name: str,
    max_items: int | None = None,
    progress_every: int = 100,
) -> np.ndarray:
    n_items = train_matrix.shape[0]
    if max_items is not None:
        n_items = min(n_items, max_items)

    h = base_exog_future.shape[0]
    preds = np.zeros((n_items, h), dtype=float)

    for i in range(n_items):
        y_train = train_matrix[i]
        if np.all(y_train == 0):
            preds[i, :] = 0.0
            continue

        y_train_series = pd.Series(y_train, index=base_exog_train.index)
        exog_train_i = base_exog_train.copy()
        exog_future_i = base_exog_future.copy()
        exog_train_i["sell_price"] = price_train[i]
        exog_future_i["sell_price"] = price_future[i]
        try:
            preds[i, :] = predict_model(
                name=model_name,
                y_train=y_train_series,
                exog_train=exog_train_i,
                exog_future=exog_future_i,
                h=h,
            )
        except Exception:
            preds[i, :] = seasonal_naive_forecast(
                y_train_series, h=h, season=CFG.seasonal_period
            )

        if progress_every and (i + 1) % progress_every == 0:
            print(f"[INFO] Processed {i + 1}/{n_items} items for {model_name}")

    return preds

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

# %%
# Main pipeline
def main():
    out_dir = ensure_dirs()
    print(f"[INFO] Project ROOT: {CFG.project_root}")
    print(f"[INFO] Outputs will be saved to: {out_dir}")
    warnings.filterwarnings("ignore", category=SMValueWarning)
    warnings.filterwarnings(
        "ignore",
        message="Could not infer format",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="No supported index is available",
        category=FutureWarning,
    )

    # 1 Load base data via EDA utilities
    calendar, sales_train_wide, sell_prices = load_data(include_id=True)
    calendar = prepare_calendar(calendar)
    calendar = add_has_event(calendar)
    calendar["date"] = pd.to_datetime(calendar["date"])

    test_valid_path = os.path.join(PROJECT_ROOT, CFG.data_dir, CFG.test_valid_file)
    test_eval_path = os.path.join(PROJECT_ROOT, CFG.data_dir, CFG.test_eval_file)
    (
        common_ids,
        train_matrix,
        valid_matrix,
        test_matrix,
        train_cols,
        valid_cols,
        test_cols,
        item_ids,
        store_ids,
    ) = prepare_item_level_matrices(
        sales_train_wide,
        test_valid_path=test_valid_path,
        test_eval_path=test_eval_path,
    )
    if CFG.focus_dept_id or CFG.focus_store_id:
        print(
            f"[INFO] Focus dept_id: {CFG.focus_dept_id}, "
            f"store_id: {CFG.focus_store_id}, items={len(common_ids)}"
        )
    else:
        print(f"[INFO] All items used: {len(common_ids)}")

    base_exog_train = exog_from_calendar_by_d(calendar, train_cols, use_range_index=True)
    base_exog_valid = exog_from_calendar_by_d(calendar, valid_cols, use_range_index=True)
    base_exog_test = exog_from_calendar_by_d(calendar, test_cols, use_range_index=True)
    price_train, price_valid, price_test = build_price_matrices(
        item_ids=item_ids,
        store_ids=store_ids,
        sell_prices=sell_prices,
        calendar_prepped=calendar,
        train_cols=train_cols,
        valid_cols=valid_cols,
        test_cols=test_cols,
    )

    if CFG.save_selection_plots and len(common_ids) > 0:
        sample_series = pd.Series(train_matrix[0], index=base_exog_train.index)
        save_acf_pacf_plot(sample_series, out_dir, filename="acf_pacf_sample_item.png")
        print("[INFO] Saved ACF/PACF plot (sample item).")

    if CFG.save_diagnostics and len(common_ids) > 0:
        sample_series = pd.Series(train_matrix[0], index=base_exog_train.index)
        exog_train_sample = base_exog_train.copy()
        exog_train_sample["sell_price"] = price_train[0]
        res_sample = fit_sarimax(
            y_train=sample_series,
            exog_train=exog_train_sample,
            order=CFG.order,
            seasonal_order=CFG.seasonal_order,
        )
        save_sarimax_diagnostics(res_sample, out_dir)
        print(f"[INFO] Saved SARIMAX diagnostics for sample item: {common_ids[0]}")

    # 2 Fast comparison on a subset of items
    compare_items = CFG.compare_max_items
    if CFG.max_items is not None:
        compare_items = (
            min(compare_items, CFG.max_items)
            if compare_items is not None
            else CFG.max_items
        )

    results_df, preds_subset = evaluate_models_item_level(
        train_matrix=train_matrix,
        valid_matrix=valid_matrix,
        base_exog_train=base_exog_train,
        base_exog_valid=base_exog_valid,
        price_train=price_train,
        price_valid=price_valid,
        models=CFG.models,
        max_items=compare_items,
        progress_every=CFG.progress_every,
    )

    comparison_path = os.path.join(out_dir, "model_comparison_validation.csv")
    results_df.to_csv(comparison_path, index=False)
    print(
        f"[VAL] Model comparison (RMSE) on {results_df['items_used'].iloc[0]} items:"
    )
    for row in results_df.itertuples(index=False):
        print(f"  {row.model}: {row.rmse:.4f}")
    print(f"[INFO] Saved model comparison -> {comparison_path}")

    best_model = results_df.iloc[0]["model"]
    print(f"[INFO] Best model from subset: {best_model}")

    full_items = len(common_ids)
    if CFG.max_items is not None:
        full_items = min(full_items, CFG.max_items)

    # 3 Full validation for the best model
    best_pred_valid_full = predict_best_model_item_level(
        train_matrix=train_matrix,
        base_exog_train=base_exog_train,
        base_exog_future=base_exog_valid,
        price_train=price_train,
        price_future=price_valid,
        model_name=best_model,
        max_items=full_items,
        progress_every=CFG.progress_every,
    )
    valid_rmse_full = rmse(
        valid_matrix[:full_items].reshape(-1),
        best_pred_valid_full.reshape(-1),
    )
    print(f"[VAL] {best_model} RMSE (full validation): {valid_rmse_full:.4f}")

    valid_dates = d_cols_to_dates(calendar, valid_cols)
    valid_actual_total = valid_matrix[:full_items].sum(axis=0)
    valid_pred_total = best_pred_valid_full.sum(axis=0)
    save_prediction_plot(
        dates=valid_dates,
        y_true=valid_actual_total,
        y_pred=valid_pred_total,
        out_dir=out_dir,
        filename=f"validation_total_actual_vs_pred_{best_model}.png",
        title=f"Validation Total: Actual vs Forecast ({best_model})",
    )
    print("[INFO] Saved validation total plot.")

    # 4 Final evaluation on test_evaluation file (item-level)
    test_pred = predict_best_model_item_level(
        train_matrix=train_matrix,
        base_exog_train=base_exog_train,
        base_exog_future=base_exog_test,
        price_train=price_train,
        price_future=price_test,
        model_name=best_model,
        max_items=full_items,
        progress_every=CFG.progress_every,
    )

    test_rmse = rmse(
        test_matrix[:full_items].reshape(-1),
        test_pred.reshape(-1),
    )
    test_out = os.path.join(out_dir, "test_predictions_item_level.csv")
    test_df = pd.DataFrame(test_pred, columns=test_cols)
    test_df.insert(0, "id", common_ids[:full_items])
    test_df.to_csv(test_out, index=False)
    print(f"[INFO] Saved test predictions -> {test_out}")
    print(f"[TEST] {best_model} RMSE on test evaluation dataset: {test_rmse:.4f}")


if __name__ == "__main__":
    main()

# %%
