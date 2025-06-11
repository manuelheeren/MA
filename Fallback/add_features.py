import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

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
    #df = add_rolling_stats(df, window=window)
    #df = add_manual_rolling_stats(df, window=window)
    return df

def add_rolling_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()
    
    # Compute predicted direction: long = 1, short = 0
    df['predicted_label'] = (df['direction'] == 'long').astype(int)

    # Compute actual direction based on pnl outcome
    def compute_true_label(row):
        if row['pnl'] > 0:
            return 1 if row['direction'] == 'long' else 0
        else:
            return 0 if row['direction'] == 'long' else 1

    df['true_label'] = df.apply(compute_true_label, axis=1)

    # Drop any rows without a valid session or label
    df = df.dropna(subset=["true_label", "session"]).reset_index(drop=True)

    stat_frames = []

    for session, group in df.groupby("session"):
        group = group.sort_values("entry_time").reset_index(drop=True)

        rolling_cols = {
            'rolling_accuracy': [],
            'rolling_f1': [],
            'rolling_precision': [],
            'rolling_recall': [],
            'rolling_auc': []
        }

        for i in range(len(group)):
            start = max(0, i - window + 1)
            y_true = group.loc[start:i, 'true_label']
            y_pred = group.loc[start:i, 'predicted_label']

            if len(y_true) < window:
                for k in rolling_cols:
                    rolling_cols[k].append(np.nan)
                continue

            try:
                auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.5
            except ValueError:
                auc = 0.5

            rolling_cols['rolling_accuracy'].append(accuracy_score(y_true, y_pred))
            rolling_cols['rolling_f1'].append(f1_score(y_true, y_pred, zero_division=0))
            rolling_cols['rolling_precision'].append(precision_score(y_true, y_pred, zero_division=0))
            rolling_cols['rolling_recall'].append(recall_score(y_true, y_pred, zero_division=0))
            rolling_cols['rolling_auc'].append(auc)

        for k in rolling_cols:
            group[k] = rolling_cols[k]

        stat_frames.append(group)

    final_df = pd.concat(stat_frames, ignore_index=True)
    return final_df

def get_binary_label(row):
    if row['status'] == 'tp_hit':
        return 1
    elif row['status'] == 'sl_hit':
        return 0
    elif row['status'] == 'session_close':
        if row['direction'] == 'long':
            return 1 if row['exit_price'] > row['entry_price'] else 0
        elif row['direction'] == 'short':
            return 1 if row['exit_price'] < row['entry_price'] else 0
    return np.nan

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

def add_manual_rolling_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["direction", "pnl", "session", "entry_time"])

    # Assign predicted label: long = 1, short = 0
    df['predicted_label'] = (df['direction'] == 'long').astype(int)

    # Define true label based on direction and outcome
    def determine_true_label(row):
        if row['direction'] == 'long':
            return 1 if row['pnl'] > 0 else 0
        elif row['direction'] == 'short':
            return 0 if row['pnl'] > 0 else 1
        return np.nan

    df['true_label'] = df.apply(determine_true_label, axis=1)
    df = df.dropna(subset=["true_label"]).reset_index(drop=True)

    stat_frames = []

    for session, group in df.groupby("session"):
        group = group.sort_values("entry_time").reset_index(drop=True)

        rolling_metrics = {
            'rolling_accuracy_manual': [],
            'rolling_precision_manual': [],
            'rolling_recall_manual': [],
            'rolling_f1_manual': [],
            'rolling_auc_manual': []
        }

        for i in range(len(group)):
            start = max(0, i - window + 1)
            y_true = group.loc[start:i, 'true_label'].values
            y_pred = group.loc[start:i, 'predicted_label'].values

            if len(y_true) < window:
                for key in rolling_metrics:
                    rolling_metrics[key].append(np.nan)
                continue

            TP = np.sum((y_pred == 1) & (y_true == 1))
            FP = np.sum((y_pred == 1) & (y_true == 0))
            TN = np.sum((y_pred == 0) & (y_true == 0))
            FN = np.sum((y_pred == 0) & (y_true == 1))

            total = TP + TN + FP + FN
            accuracy = (TP + TN) / total if total > 0 else np.nan
            precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
            recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan

            try:
                auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.5
            except:
                auc = 0.5

            rolling_metrics['rolling_accuracy_manual'].append(accuracy)
            rolling_metrics['rolling_precision_manual'].append(precision)
            rolling_metrics['rolling_recall_manual'].append(recall)
            rolling_metrics['rolling_f1_manual'].append(f1)
            rolling_metrics['rolling_auc_manual'].append(auc)

        for key in rolling_metrics:
            group[key] = rolling_metrics[key]

        stat_frames.append(group)

    final_df = pd.concat(stat_frames, ignore_index=True)
    return final_df