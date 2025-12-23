"""
AFCS 2025 - Sales Time Series Forecasting (TX3 / Food3)
EDA Script
"""
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.style.use("default")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CALENDAR_PATH = os.path.join(DATA_DIR, "calendar_afcs2025.csv")
SALES_TRAIN_PATH = os.path.join(DATA_DIR, "sales_train_validation_afcs2025.csv")
SELL_PRICES_PATH = os.path.join(DATA_DIR, "sell_prices_afcs2025.csv")
EDA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "eda_outputs")
SAVE_EDA = True
SHOW_PLOTS = False

# Light EDA for use
FOCUS_STORE_ID = "TX_3"
FOCUS_CAT_ID = None
FOCUS_DEPT_ID = "FOODS_3"
REPORT_TAG = "tx3_foods3"
SAVE_LEGACY_NAMES = True
SAVE_REPORT_NAMES = False
SAVE_COMBINED_EVENT_SNAP = False


def report_name(stem: str) -> str:
    return f"{stem}_{REPORT_TAG}.png"


def finalize_plot(legacy_name: str | None = None, report_stem: str | None = None, fig=None) -> None:
    fig = fig or plt.gcf()
    if SAVE_EDA:
        os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
        filenames: list[str] = []
        if SAVE_LEGACY_NAMES and legacy_name:
            filenames.append(legacy_name)
        if SAVE_REPORT_NAMES and report_stem:
            filenames.append(report_name(report_stem))
        for filename in filenames:
            fig.savefig(os.path.join(EDA_OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# 1. LOAD DATA

def load_data(include_id: bool = False):
    print("Loading data...")
    calendar = pd.read_csv(CALENDAR_PATH)
    sales_train = pd.read_csv(SALES_TRAIN_PATH)
    sell_prices = pd.read_csv(SELL_PRICES_PATH)

    parts = sales_train["id"].str.split("_", expand=True)
    sales_train["cat_id"] = parts[0]
    sales_train["dept_id"] = parts[0] + "_" + parts[1] 
    sales_train["item_id"] = parts[0] + "_" + parts[1] + "_" + parts[2] 
    sales_train["state_id"] = parts[3] 
    sales_train["store_id"] = parts[3] + "_" + parts[4]

    day_cols = [c for c in sales_train.columns if c.startswith("d_")]
    base_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    if include_id:
        base_cols = ["id"] + base_cols

    sales_train = sales_train[base_cols + day_cols]

    print("\n--- dataset shapes ---")
    print(f"calendar: {calendar.shape[0]} rows, {calendar.shape[1]} cols")
    print(f"sales_train: {sales_train.shape[0]} rows, {sales_train.shape[1]} cols")
    print(f"sell_prices: {sell_prices.shape[0]} rows, {sell_prices.shape[1]} cols")

    return calendar, sales_train, sell_prices

# 2. PREPROCESS & MERGE

def prepare_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure calendar has:
    - 'date' as datetime
    - 'd' column matching 'd_1', 'd_2', ... (if not present, create it)
    """
    calendar = calendar.copy()

    # Convert date to datetime
    if 'date' in calendar.columns:
        calendar['date'] = pd.to_datetime(calendar['date'])
    else:
        raise ValueError("calendar csv must have a 'date' column.")

    # Ensure 'd' column exists
    if 'd' not in calendar.columns:
        calendar = calendar.sort_values('date').reset_index(drop=True)
        calendar['d'] = 'd_' + (calendar.index + 1).astype(str)
    else:
        calendar['d'] = calendar['d'].astype(str)

    return calendar


def melt_sales_train(sales_train: pd.DataFrame) -> pd.DataFrame:
    
    id_vars = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    value_vars = [c for c in sales_train.columns if c.startswith('d_')]

    print(f"\nMelting sales_train: {len(value_vars)} daily columns detected.")
    sales_long = sales_train.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='d',
        value_name='sales'
    )

    print("Shape after melt:", sales_long.shape)
    return sales_long

def merge_all(calendar: pd.DataFrame,
              sales_long: pd.DataFrame,
              sell_prices: pd.DataFrame) -> pd.DataFrame:
    
    print("\nMerging sales with calendar...")
    df = sales_long.merge(calendar, on='d', how='left')

    # Basic sanity checks
    print("Merged with calendar, shape:", df.shape)
    missing_dates = df['date'].isna().mean()
    print(f"Fraction of missing date after merge: {missing_dates:.4f}")

    # Merge with prices
    print("Merging with sell_prices...")
    df = df.merge(
        sell_prices,
        on=['store_id', 'item_id', 'wm_yr_wk'],
        how='left'
    )
    print("Merged with prices, shape:", df.shape)
    print(f"Fraction of missing sell_price: {df['sell_price'].isna().mean():.4f}")

    return df

# TX3 / Food3 specific optional filtering
def optional_filter_focus(df: pd.DataFrame) -> pd.DataFrame:
    
    df_filtered = df.copy()

    if FOCUS_STORE_ID is not None:
        print(f"\nFiltering to store_id == {FOCUS_STORE_ID}")
        df_filtered = df_filtered[df_filtered['store_id'] == FOCUS_STORE_ID]

    if FOCUS_CAT_ID is not None:
        print(f"Filtering to cat_id == {FOCUS_CAT_ID}")
        df_filtered = df_filtered[df_filtered['cat_id'] == FOCUS_CAT_ID]

    if FOCUS_DEPT_ID is not None and 'dept_id' in df_filtered.columns:
        print(f"Filtering to dept_id == {FOCUS_DEPT_ID}")
        df_filtered = df_filtered[df_filtered['dept_id'] == FOCUS_DEPT_ID]

    print("Shape after optional filter:", df_filtered.shape)
    return df_filtered


# 3. FORECASTING-ORIENTED EDA

def plot_overall_trend(df: pd.DataFrame):
    
    print("\n[EDA] Overall sales trend...")

    daily_total = df.groupby('date')['sales'].sum().sort_index()

    plt.figure(figsize=(14, 5))
    plt.plot(daily_total.index, daily_total.values, label='Daily total sales', alpha=0.5)
    plt.plot(daily_total.rolling(7).mean(), label='7-day rolling mean')
    plt.plot(daily_total.rolling(28).mean(), label='28-day rolling mean')
    plt.title("Total daily sales over time")
    plt.xlabel("Date")
    plt.ylabel("Units sold")
    plt.legend()
    plt.tight_layout()
    finalize_plot(legacy_name="overall_trend.png", report_stem="overall_trend")


def plot_weekly_seasonality(df: pd.DataFrame):
    
    print("\n[EDA] Weekly seasonality (weekday pattern)...")

    # Ensure weekday column exists
    if 'weekday' not in df.columns:
        print("No 'weekday' column in df; skipping weekly seasonality plot.")
        return

    weekday_order = ['Saturday', 'Sunday', 'Monday', 'Tuesday',
                     'Wednesday', 'Thursday', 'Friday']
    weekday_mean = df.groupby('weekday')['sales'].mean().reindex(weekday_order)

    plt.figure(figsize=(8, 4))
    weekday_mean.plot(kind='bar')
    plt.title("Average sales by weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Average units sold")
    plt.tight_layout()
    finalize_plot(legacy_name="weekly_seasonality.png", report_stem="weekday_effects")


def plot_monthly_seasonality(df: pd.DataFrame):
    
    print("\n[EDA] Monthly seasonality...")

    if 'month' not in df.columns:
        print("No 'month' column in df; skipping monthly seasonality plot.")
        return

    month_mean = df.groupby('month')['sales'].mean().sort_index()

    plt.figure(figsize=(8, 4))
    month_mean.plot(kind='bar')
    plt.title("Average sales by month")
    plt.xlabel("Month")
    plt.ylabel("Average units sold")
    plt.tight_layout()
    finalize_plot(legacy_name="monthly_seasonality.png", report_stem="monthly_seasonality")


def plot_event_and_snap_effects(df: pd.DataFrame):
    
    print("\n[EDA] Event & SNAP effects...")

    df = df.copy()

    # Event indicator: if there is any event_name_1 or event_name_2
    df['has_event'] = (~df['event_name_1'].isna()) | (~df['event_name_2'].isna())
    event_mean = df.groupby('has_event')['sales'].mean()

    plt.figure(figsize=(6, 4))
    event_mean.index = ['No event', 'Has event']
    event_mean.plot(kind='bar')
    plt.title("Average sales: event vs non-event days")
    plt.ylabel("Average units sold")
    plt.tight_layout()
    finalize_plot(legacy_name="event_effect.png", report_stem="event_effect")

    # SNAP effect 
    if 'snap_TX' in df.columns:
        snap_mean = df.groupby('snap_TX')['sales'].mean()

        plt.figure(figsize=(6, 4))
        snap_mean.index = ['Non-SNAP days', 'SNAP days']
        snap_mean.plot(kind='bar')
        plt.title("Average sales: SNAP vs non-SNAP days (TX)")
        plt.ylabel("Average units sold")
        plt.tight_layout()
        finalize_plot(legacy_name="snap_effect.png", report_stem="snap_effect")
    else:
        print("No 'snap_TX' column; skipping SNAP effect plot.")

    # Combined SNAP + event comparison for report
    if SAVE_COMBINED_EVENT_SNAP and 'snap_TX' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        event_mean.plot(kind='bar', ax=axes[0], color="tab:blue")
        axes[0].set_title("Event vs non-event")
        axes[0].set_ylabel("Average units sold")
        axes[0].set_xlabel("")

        snap_mean = df.groupby('snap_TX')['sales'].mean()
        snap_mean.index = ['Non-SNAP days', 'SNAP days']
        snap_mean.plot(kind='bar', ax=axes[1], color="tab:green")
        axes[1].set_title("SNAP vs non-SNAP")
        axes[1].set_xlabel("")
        fig.tight_layout()
        finalize_plot(report_stem="snap_event_comparison", fig=fig)


def stl_decomposition(df: pd.DataFrame):
    
    print("\n[EDA] STL decomposition on aggregate series...")

    daily_total = df.groupby('date')['sales'].sum().sort_index()

    # For STL, choose a seasonal period. Retail usually has weekly seasonality => period=7
    period = 7

    # Drop zeros-only early part if needed, but here we just pass the full series.
    stl = STL(daily_total, period=period, robust=True)
    result = stl.fit()

    fig = result.plot()
    fig.set_size_inches(10, 8)
    plt.tight_layout()
    finalize_plot(legacy_name="stl_decomposition.png", report_stem="stl_decomposition", fig=fig)


def stationarity_adf_test(df: pd.DataFrame):
    
    print("\n[EDA] Stationarity test (ADF) on aggregate series...")

    daily_total = df.groupby('date')['sales'].sum().sort_index()

    result = adfuller(daily_total)
    adf_stat, pvalue, usedlag, nobs, crit_values, icbest = result

    print("ADF Statistic: {:.4f}".format(adf_stat))
    print("p-value: {:.4f}".format(pvalue))
    print("Critical values:")
    for k, v in crit_values.items():
        print(f"   {k}: {v:.4f}")

    if pvalue < 0.05:
        print("=> Series is likely stationary (reject H0).")
    else:
        print("=> Series is likely non-stationary (fail to reject H0).")


def plot_acf_pacf_aggregate(df: pd.DataFrame, nlags: int = 30):
    
    print("\n[EDA] ACF & PACF for aggregate series...")

    daily_total = df.groupby('date')['sales'].sum().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(daily_total, lags=nlags, ax=axes[0])
    plot_pacf(daily_total, lags=nlags, ax=axes[1])
    axes[0].set_title("ACF of total daily sales")
    axes[1].set_title("PACF of total daily sales")
    plt.tight_layout()
    finalize_plot(legacy_name="acf_pacf_aggregate.png", report_stem="acf_pacf_aggregate", fig=fig)


def item_level_variability(df: pd.DataFrame):
    
    print("\n[EDA] Item-level variability (mean vs variance)...")

    item_stats = df.groupby('item_id')['sales'].agg(['mean', 'var']).reset_index()
    item_stats = item_stats.replace([np.inf, -np.inf], np.nan).dropna()

    plt.figure(figsize=(6, 5))
    plt.scatter(item_stats['mean'], item_stats['var'], alpha=0.3)
    plt.xlabel("Item mean sales")
    plt.ylabel("Item sales variance")
    plt.title("Mean–variance relationship across items")
    plt.tight_layout()
    finalize_plot(legacy_name="item_mean_variance.png", report_stem="mean_variance_scatter")

    print(item_stats.describe())


def price_dynamics(df: pd.DataFrame):
    
    print("\n[EDA] Price dynamics & relation to sales...")

    if 'sell_price' not in df.columns:
        print("No 'sell_price' column; skipping price analysis.")
        return

    # Basic price distribution
    plt.figure(figsize=(6, 4))
    df['sell_price'].dropna().hist(bins=50)
    plt.title("Sell price distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    finalize_plot(legacy_name="price_distribution.png", report_stem="price_distribution")

    # Sample one item for time series analysis
    sample_item_id = df['item_id'].value_counts().index[0]
    print("Using sample item for price & sales time series:", sample_item_id)

    sample_df = df[df['item_id'] == sample_item_id].copy()
    sample_df = sample_df.sort_values('date')

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(sample_df['date'], sample_df['sales'], label='Sales', alpha=0.7)
    ax1.set_ylabel("Units sold")
    ax2 = ax1.twinx()
    ax2.plot(sample_df['date'], sample_df['sell_price'], label='Price', alpha=0.7)
    ax2.set_ylabel("Price")
    plt.title(f"Sales and price over time for item {sample_item_id}")
    fig.tight_layout()
    finalize_plot(legacy_name="price_sales_timeseries.png", fig=fig)

    # Scatter sales vs price (sample subset to reduce noise)
    sample_for_scatter = df[['sell_price', 'sales']].dropna().sample(
        min(10000, df['sell_price'].dropna().shape[0]),
        random_state=42
    )
    plt.figure(figsize=(6, 5))
    plt.scatter(sample_for_scatter['sell_price'],
                sample_for_scatter['sales'],
                alpha=0.2)
    plt.xlabel("Price")
    plt.ylabel("Units sold")
    plt.title("Sales vs price (sample)")
    plt.tight_layout()
    finalize_plot(legacy_name="price_sales_scatter.png")


def hierarchical_structure(df: pd.DataFrame):
    
    print("\n[EDA] Hierarchical structure (category / department / item)...")

    # Category-level share
    if 'cat_id' in df.columns:
        cat_total = df.groupby('cat_id')['sales'].sum().sort_values(ascending=False)
        plt.figure(figsize=(6, 4))
        cat_total.plot(kind='bar')
        plt.title("Total sales by category")
        plt.ylabel("Total units sold")
        plt.tight_layout()
        finalize_plot(legacy_name="category_sales.png")
        print("\nCategory-level total sales:")
        print(cat_total)

    # Department-level share
    if 'dept_id' in df.columns:
        dept_total = df.groupby('dept_id')['sales'].sum().sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        dept_total.plot(kind='bar')
        plt.title("Total sales by department")
        plt.ylabel("Total units sold")
        plt.tight_layout()
        finalize_plot(legacy_name="department_sales.png")
        print("\nDepartment-level total sales:")
        print(dept_total)

    # Store-level share
    if 'store_id' in df.columns:
        store_total = df.groupby('store_id')['sales'].sum().sort_values(ascending=False)
        plt.figure(figsize=(6, 4))
        store_total.plot(kind='bar')
        plt.title("Total sales by store")
        plt.ylabel("Total units sold")
        plt.tight_layout()
        finalize_plot(legacy_name="store_sales.png")
        print("\nStore-level total sales:")
        print(store_total)



# %% 
def main():
    calendar, sales_train, sell_prices = load_data()
    calendar = prepare_calendar(calendar)
    sales_long = melt_sales_train(sales_train)
    df = merge_all(calendar, sales_long, sell_prices)
    df = optional_filter_focus(df)

    plot_overall_trend(df)
    plot_weekly_seasonality(df)
    plot_monthly_seasonality(df)
    plot_event_and_snap_effects(df)
    stl_decomposition(df)
    plot_acf_pacf_aggregate(df)

    item_level_variability(df) 
    price_dynamics(df)        
    hierarchical_structure(df) 

if __name__ == "__main__":
    main()
