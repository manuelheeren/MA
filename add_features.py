import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def generate_meta_labeled_data(trade_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = trade_df.copy()

    # Assign meta labels
    def assign_meta_label(row):
        if row['status'] == 'session_close':
            if row['direction'] == 'long':
                return 1 if row['exit_price'] > row['entry_price'] else 0
            elif row['direction'] == 'short':
                return 1 if row['exit_price'] < row['entry_price'] else 0
        elif row['status'] == 'tp_hit':
            return 1
        elif row['status'] == 'sl_hit':
            return 0
        return np.nan

    df['meta_label'] = df.apply(assign_meta_label, axis=1)

    # Add rolling evaluation stats
    df = add_rolling_stats(df, window=window)
    return df

def add_rolling_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values("entry_time").reset_index(drop=True)

    # Rolling average PnL
    df[f'rolling_pnl_{window}'] = df['pnl'].rolling(window=window, min_periods=1).mean()

    # Rolling win rate
    df[f'rolling_winrate_{window}'] = (
        (df['pnl'] > 0).astype(int).rolling(window=window, min_periods=1).mean()
    )

    # Rolling standard deviation of PnL (volatility)
    df[f'rolling_volatility_{window}'] = df['pnl'].rolling(window=window, min_periods=1).std()

    return df


def merge_with_raw_features(trade_df: pd.DataFrame, asset: str) -> pd.DataFrame:
    feature_path = Path(f"data/processed/{asset}/combined_data.csv")
    if not feature_path.exists():
        print(f"[WARNING] Raw feature file not found for {asset}, skipping merge.")
        return trade_df

    raw_features = pd.read_csv(feature_path, parse_dates=["timestamp"])

    # Rename timestamp to entry_time to match trade_df
    raw_features = raw_features.rename(columns={"timestamp": "entry_time"})

    # Ensure both timestamps are UTC-aware and floored to the same granularity
    trade_df['entry_time'] = pd.to_datetime(trade_df['entry_time']).dt.tz_convert("UTC").dt.floor("min")
    raw_features['entry_time'] = pd.to_datetime(raw_features['entry_time']).dt.tz_convert("UTC").dt.floor("min")

    # Select only columns needed for merge
    selected_columns = [
        "entry_time","atr_14", "ma_14", "ma_30", "ma_100",
        "day_of_week", "week_number",
        "max_price_14", "min_price_14",
        "max_price_30", "min_price_30",
        "max_price_100", "min_price_100"
    ]
    missing = [col for col in selected_columns if col not in raw_features.columns]
    if missing:
        print(f"[WARNING] Columns {missing} not found in {feature_path.name}, skipping merge.")
        return trade_df

    raw_features = raw_features[selected_columns]

    # Standard merge on matching entry_time values only
    merged = pd.merge(trade_df, raw_features, on="entry_time", how="left")

    return merged