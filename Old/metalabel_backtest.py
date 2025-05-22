# metalabel_backtest.py

import pandas as pd
from pathlib import Path
import logging
from new_strategy import Asset, BetSizingMethod, get_bet_sizing
import nbimporter
from backtest import Backtest
from meta_strategy import MetaLabelingStrategy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------- MetaModelHandler ---------------------- #
class MetaModelHandler:
    def __init__(self):
        self.long_model = None
        self.short_model = None
        self.long_scaler = None
        self.short_scaler = None
        self.feature_cols = []

    def train(self, trades_df: pd.DataFrame, feature_cols: list):
        self.feature_cols = feature_cols
        trades_df = trades_df.dropna(subset=feature_cols + ['meta_label'])

        long_trades = trades_df[trades_df['direction'] == 'long']
        short_trades = trades_df[trades_df['direction'] == 'short']

        def preprocess(df):
            X = df[feature_cols]
            y = df['meta_label']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            return X_scaled, y, scaler

        X_long, y_long, self.long_scaler = preprocess(long_trades)
        X_short, y_short, self.short_scaler = preprocess(short_trades)

        self.long_model = CalibratedClassifierCV(LogisticRegression(), method='sigmoid').fit(X_long, y_long)
        self.short_model = CalibratedClassifierCV(LogisticRegression(), method='sigmoid').fit(X_short, y_short)

    def is_trade_approved(self, features: dict, direction: str, threshold: float = 0.6) -> bool:
        df = pd.DataFrame([features])[self.feature_cols]
        if direction == 'long':
            X = self.long_scaler.transform(df)
            prob = self.long_model.predict_proba(X)[0, 1]
        else:
            X = self.short_scaler.transform(df)
            prob = self.short_model.predict_proba(X)[0, 1]
        return prob >= threshold

# ---------------------- Helper Functions ---------------------- #
def train_test_split_time_series(trades_df: pd.DataFrame, test_size: float = 0.3):
    trades_df = trades_df.sort_values('entry_time')
    split_idx = int(len(trades_df) * (1 - test_size))
    return trades_df.iloc[:split_idx], trades_df.iloc[split_idx:]

def load_meta_labeled_data(asset: Asset, method: BetSizingMethod, test_size=0.3):
    meta_path = Path(f"data/metalabels/meta_labels_{asset.value}_{method.value}.csv")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta-labeled file not found: {meta_path}")

    df = pd.read_csv(meta_path, parse_dates=["entry_time", "exit_time"])
    df = df.dropna(subset=["meta_label"])
    df = df.sort_values("entry_time")
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]

def train_meta_model(train_df: pd.DataFrame, feature_cols: list) -> MetaModelHandler:
    meta_model = MetaModelHandler()
    meta_model.train(train_df, feature_cols)
    return meta_model

# ---------------------- Run Backtest ---------------------- #
def run_metalabel_backtest(asset: Asset, method: BetSizingMethod, feature_cols: list):
    # Load raw price data
    price_data_path = Path(f"data/processed/{asset.value}/combined_data.csv")
    price_data = pd.read_csv(price_data_path, index_col="timestamp", parse_dates=True)

    # Load and split meta-labeled data
    train_df, test_df = load_meta_labeled_data(asset, method)
    test_start = test_df['entry_time'].min()

    # Train meta model on training data
    meta_handler = train_meta_model(train_df, feature_cols)

    # Restrict price data to test period (to simulate live trading post-training)
    price_data = price_data[price_data.index >= test_start]

    # Get bet sizing method
    past_returns = price_data['close'].pct_change().dropna()
    bet_sizing = get_bet_sizing(method, past_returns)

    # Run strategy with meta-model
    strategy = MetaLabelingStrategy(price_data, asset.value, bet_sizing, method, meta_model_handler=meta_handler)
    strategy.generate_signals()
    strategy.simulate_trades()

    # Backtest results
    backtest = Backtest(strategy)
    backtest.run_analysis()
    backtest.print_summary()

    # Save filtered trades to CSV for inspection
    output_dir = Path("data/results_metalabel")
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"trades_meta_filtered_{asset.value}_{method.value}.csv"
    filtered_trades_df = strategy.get_trade_data()
    filtered_trades_df.to_csv(output_dir / filename, index=False)

    logging.info(f"Saved filtered trades to: {output_dir / filename}")

# --- Print key performance metrics ---
    metrics = backtest.results['sessions']
    print(f"\n--- Meta-Filtered Performance Summary ({asset.value}, {method.value}) ---")
    for session, result in metrics.items():
        m = result['metrics']
        print(f"\n[{session.upper()} SESSION]")
        print(f"Total PnL: ${m.total_pnl:,.2f}")
        print(f"Return: {m.total_return_pct:.2f}%")
        print(f"Sharpe Ratio: {m.sharpe:.2f}" if m.sharpe else "Sharpe: n/a")
        print(f"Win Rate: {m.win_rate:.2%}")
        print(f"Max Drawdown: {m.max_drawdown_pct:.2f}%")


# ---------------------- Main Entry ---------------------- #
if __name__ == "__main__":
    asset = Asset.XAUUSD
    method = BetSizingMethod.OPTIMAL_F
    feature_cols = ["atr_value", "attempt", "duration_minutes", "day_of_week", "ref_close"]

    run_metalabel_backtest(asset, method, feature_cols)
