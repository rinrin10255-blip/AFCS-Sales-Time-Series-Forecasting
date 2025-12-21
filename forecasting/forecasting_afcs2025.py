# %%
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

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


def fit_sarimax_and_forecast(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    exog_future: pd.DataFrame,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    h: int
) -> np.ndarray:
    y = y_train.astype("float64").copy()
    y_log = np.log1p(y)
    X_train = exog_train.astype("float64")
    X_future = exog_future.astype("float64")

    model = SARIMAX(
        y_log,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=200)
    fcast = res.get_forecast(steps=h, exog=X_future)

    yhat_log = fcast.predicted_mean
    yhat = np.expm1(yhat_log).values

    yhat = np.clip(yhat, 0, None)

    return yhat

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

    # 3 Validation on official test_validation file (aggregate)
    y_valid, sarimax_pred_valid = forecast_on_wide_file(
        y_train_all=y_all,
        calendar=calendar,
        file_name=CFG.test_valid_file,
    )
    h = len(y_valid)
    naive_pred_valid = seasonal_naive_forecast(y_all, h=h, season=CFG.seasonal_period)
    naive_rmse = rmse(y_valid.values, naive_pred_valid)
    sarimax_rmse = rmse(y_valid.values, sarimax_pred_valid)
    print(f"[VAL] Seasonal-Naive RMSE (official validation): {naive_rmse:.4f}")
    print(f"[VAL] SARIMAX RMSE (official validation): {sarimax_rmse:.4f}")

    # Validation predictions are not saved to disk for this project.

    # 4 Final evaluation on test_evaluation file (aggregate)
    y_test, sarimax_pred_test = forecast_on_wide_file(
        y_train_all=y_all,
        calendar=calendar,
        file_name=CFG.test_eval_file,
    )
    test_rmse = rmse(y_test.values, sarimax_pred_test)
    print(f"[TEST] SARIMAX RMSE on test evaluation dataset: {test_rmse:.4f}")

    # 5 No submission file is generated for this project.


if __name__ == "__main__":
    main()

# %%
