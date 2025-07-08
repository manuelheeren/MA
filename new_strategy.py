from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import time
import logging
from enum import Enum
from bet_sizing import KellyBetSizing, FixedFractionalBetSizing, BetSizingStrategy, FixedBetSize, PercentVolatilityBetSizing, OptimalF
from collections import deque
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Configure logging OLD
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Asset(Enum):
    XAUUSD = "XAUUSD"
    BTCUSD = "BTCUSD"
    SPYUSD = "SPYUSD"
    WTI = "WTI"

class BetSizingMethod(Enum):
    KELLY = "kelly"
    FIXED = "fixed"
    FIXED_AMOUNT = "fixed_amount"
    PERCENT_VOLATILITY = "percent_volatility"
    OPTIMAL_F = "optimal_f"

@dataclass(frozen=True)
class SessionTime:
    name: str
    start: time
    end: time
    close: time

@dataclass
class TradeSetup:
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    attempt: int
    ref_close: float
    position_size: float
    risk_amount: float
    session: str
    atr_14: Optional[float] = None
    ma_14: Optional[float] = None
    min_price_30: Optional[float] = None
    max_price_30: Optional[float] = None

@dataclass
class Trade:
    entry_time: pd.Timestamp
    setup: TradeSetup
    session: str
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    status: str = 'open'
    pnl: Optional[float] = None
    equity_at_entry: Optional[float] = None

    @property
    def holding_time(self) -> Optional[pd.Timedelta]:
        return self.exit_time - self.entry_time if self.exit_time else None

    @property
    def return_pct(self) -> Optional[float]:
        if (
            self.pnl is not None
            and self.setup.risk_amount
            and np.isfinite(self.setup.risk_amount)
            and self.setup.risk_amount != 0
        ):
            return (self.pnl / self.setup.risk_amount) * 0.01
        return None

class TradingStrategy:

    SESSIONS = [
        SessionTime('asian', time(0, 0), time(8, 0), time(7, 59)),
        SessionTime('london', time(8, 0), time(16, 0), time(15, 59)),
        SessionTime('us', time(13, 0), time(21, 0), time(20, 59))
    ]

    SESSION_MAP = {'asian': 0, 'london': 1, 'us': 2}

    MAX_ATTEMPTS = 3
    INITIAL_CAPITAL = 100000

    def __init__(self, data: pd.DataFrame, asset: str, bet_sizing: BetSizingStrategy, bet_sizing_method: BetSizingMethod, rolling_window=30):
        self.data = data
        self.rolling_metrics = {
            'asian': RollingMetrics(window_size=rolling_window),
            'london': RollingMetrics(window_size=rolling_window),
            'us': RollingMetrics(window_size=rolling_window)
        }
        self.asset = Asset(asset)
        self.bet_sizing = bet_sizing
        self.bet_sizing_method = bet_sizing_method
        self.session_capital = {s.name: self.INITIAL_CAPITAL for s in self.SESSIONS}
        self.trades = {s.name: [] for s in self.SESSIONS}
        logger.info(f"Strategy initialized for {self.asset.value} using {type(bet_sizing).__name__}")

    def _calculate_trade_levels(self, price: float, direction: str, attempt: int) -> tuple:
        sl_pct = 0.005
        tp_pct = sl_pct * attempt
        if direction == 'long':
            stop_loss = price * (1 - sl_pct)
            take_profit = price * (1 + tp_pct)
        else:
            stop_loss = price * (1 + sl_pct)
            take_profit = price * (1 - tp_pct)
        return stop_loss, take_profit

    def _create_trade_setup(
        self,
        entry_time: pd.Timestamp,
        price: float,
        direction: str,
        attempt: int,
        ref_close: float,
        session: str
    ) -> TradeSetup:
        stop_loss, take_profit = self._calculate_trade_levels(price, direction, attempt)
        current_equity = self._get_session_equity(session, price)
        available_cash = self._get_session_available_cash(session)

        # Extract all needed features from self.data (safe access)
        price_features = ["ma_14", "min_price_30", "max_price_30", "atr_14","daily_return","daily_volatility","t10yie","vix_close"]
        context = {}

        if entry_time in self.data.index:
            for col in price_features:
                context[col] = self.data.at[entry_time, col]
        else:
            for col in price_features:
                context[col] = None  # fallback for missing

        # Add trade-specific info to context
        metrics = self.rolling_metrics[session].latest()

        session_map = {'asian': 0, 'london': 1, 'us': 2}
        session_code = session_map.get(session, -1)

        context.update({
            "attempt": attempt,
            "ref_close": ref_close,
            "duration_minutes": 0,
            "session": session,
            "rolling_f1": metrics.get("rolling_f1"),
            "rolling_accuracy": metrics.get("rolling_accuracy"),
            "rolling_precision": metrics.get("rolling_precision"),
            "rolling_recall": metrics.get("rolling_recall"),
            "n_total_seen": metrics.get("n_total_seen"),
            "n_window_obs": metrics.get("n_window_obs"),
            "session_code": session_code,
        })

        # Call the bet sizing strategy
        result = self.bet_sizing.compute_position(
            equity=current_equity,
            price=price,
            stop_loss=stop_loss,
            available_cash=available_cash,
            context=context,
            session=session
        )

        # ✅ Handle both 2-value and 3-value return styles safely
        if isinstance(result, tuple):
            if len(result) == 3:
                position_size, risk_amount, enriched_context = result
            elif len(result) == 2:
                position_size, risk_amount = result
                enriched_context = context
            else:
                raise ValueError(f"Unexpected number of return values from compute_position: {len(result)}")
        else:
            raise ValueError("compute_position must return a tuple")

        # Extract features (fallback to None if not found)
        atr_14 = enriched_context.get("atr_14")
        ma_14 = enriched_context.get("ma_14")
        min_price_30 = enriched_context.get("min_price_30")
        max_price_30 = enriched_context.get("max_price_30")

        print("📦 Creating TradeSetup with:")
        print(f"  entry_time: {entry_time}")
        print(f"  atr_14: {atr_14}, ma_14: {ma_14}, min_price_30: {min_price_30}, max_price_30: {max_price_30}")
        print(f"  context keys: {list(context.keys())}")

 
        # Return the TradeSetup object
        return TradeSetup(
            direction=direction,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            attempt=attempt,
            ref_close=ref_close,
            position_size=position_size,
            risk_amount=risk_amount,
            session=session,
            atr_14=atr_14,
            ma_14=ma_14,
            min_price_30=min_price_30,
            max_price_30=max_price_30
        )

    def generate_signals(self) -> None:
        self.trade_signals = {s.name: [] for s in self.SESSIONS}
        for date in sorted(self.data['date'].unique()):
            if pd.Timestamp(date).weekday() >= 5:
                continue
            for session in self.SESSIONS:
                session_start = pd.Timestamp(f"{date} {session.start}", tz='UTC')
                if session_start not in self.data.index:
                    continue
                prev_close = self._get_previous_session_close(session_start, session)
                if not prev_close:
                    continue
                current_price = self.data.loc[session_start, 'close']
                direction = 'long' if current_price > prev_close else 'short'
                self.trade_signals[session.name].append({
                    'entry_time': session_start,
                    'direction': direction,
                    'ref_close': prev_close,
                    'session_name': session.name  # Add session info
                })

    def simulate_trades(self) -> None:
        # Flatten all signals from all sessions ('asian', 'london', 'us')
        all_signals = []
        for session_signals in self.trade_signals.values():
            all_signals.extend(session_signals)

        # Sort signals chronologically
        all_signals.sort(key=lambda x: x['entry_time'])

        # Process each signal safely
        for signal in all_signals:
            entry_time = signal['entry_time']

            if entry_time not in self.data.index:
                print(f"❌ entry_time {entry_time} not in data index — skipping signal.")
                continue

            self._process_single_signal(signal)


    def _process_single_signal(self, signal: dict) -> None:
        entry_time = signal['entry_time']
        session_name = signal['session_name']
        session = next(s for s in self.SESSIONS if s.name == session_name)
        session_end = pd.Timestamp(f"{entry_time.date()} {session.end}", tz='UTC')
        price = self.data.loc[entry_time, 'close']

        # 🛡️ Try to create a trade setup
        setup = self._create_trade_setup(
            entry_time=entry_time,
            price=price,
            direction=signal['direction'],
            attempt=1,
            ref_close=signal['ref_close'],
            session=session_name
        )

        if setup is None:
            # Trade was rejected by meta-model or missing data
            return

        equity_at_entry = self._get_session_equity(session_name, price)
        trade = Trade(entry_time, setup, session_name, equity_at_entry=equity_at_entry)
        remaining_trades = []

        self._process_single_trade(
            trade,
            self._get_session_prices(entry_time, session_end),
            session_end,
            remaining_trades
        )
        self.trades[session_name].append(trade)

        while remaining_trades:
            re_trade = remaining_trades.pop(0)
            self._process_single_trade(
                re_trade,
                self._get_session_prices(re_trade.entry_time, session_end),
                session_end,
                remaining_trades
            )
            self.trades[session_name].append(re_trade)

    def _process_single_trade(self, trade: Trade, prices: pd.DataFrame, session_end: pd.Timestamp, trades_to_process: List[Trade]) -> bool:
        for timestamp, price_data in prices.iterrows():
            if self._check_take_profit(trade, price_data):
                self._close_trade(trade, timestamp, trade.setup.take_profit, 'tp_hit')
                return True
            if self._check_stop_loss(trade, price_data):
                self._close_trade(trade, timestamp, trade.setup.stop_loss, 'sl_hit')

                # Check if a re-entry is allowed
                if trade.setup.attempt < self.MAX_ATTEMPTS and timestamp < session_end:
                    new_price = trade.setup.stop_loss
                    new_entry_time = timestamp

                    # Create new trade setup for retry
                    new_setup = self._create_trade_setup(
                        entry_time=new_entry_time,
                        price=new_price,
                        direction=trade.setup.direction,
                        attempt=trade.setup.attempt + 1,
                        ref_close=trade.setup.ref_close,
                        session=trade.session
                    )

                    if new_setup is None:
                        print(f"⛔ Retry trade rejected by meta model at {new_entry_time}")
                        return True  # Don't proceed with this retry
                        
                    # FIX: Set equity_at_entry for the new trade attempt
                    equity_at_entry = self._get_session_equity(trade.session, new_price)

                    # Append re-entry trade to processing queue
                    trades_to_process.append(Trade(
                        entry_time=new_entry_time,
                        setup=new_setup,
                        session=trade.session,
                        equity_at_entry=equity_at_entry
                    ))
                return True

        # If session ends and neither TP nor SL hit
        last_price = prices.iloc[-1]['close'] if not prices.empty else trade.setup.entry_price
        self._close_trade(trade, session_end, last_price, 'session_close')
        return True


    def _get_session_prices(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
        mask = (self.data.index > start_time) & (self.data.index <= end_time)
        return self.data[mask]

    def _get_previous_session_close(self, current_time: pd.Timestamp, session: SessionTime) -> Optional[float]:
        prev_close_time = (current_time.normalize() - pd.Timedelta(days=1)).replace(
            hour=session.close.hour,
            minute=session.close.minute
        )
        prev_data = self.data[self.data.index <= prev_close_time]
        return prev_data.iloc[-1]['close'] if not prev_data.empty else None

    def _check_take_profit(self, trade: Trade, price_data: pd.Series) -> bool:
        return ((trade.setup.direction == 'long' and price_data['high'] >= trade.setup.take_profit) or
                (trade.setup.direction == 'short' and price_data['low'] <= trade.setup.take_profit))

    def _check_stop_loss(self, trade: Trade, price_data: pd.Series) -> bool:
        return ((trade.setup.direction == 'long' and price_data['low'] <= trade.setup.stop_loss) or
                (trade.setup.direction == 'short' and price_data['high'] >= trade.setup.stop_loss))

    def _close_trade(self, trade: Trade, exit_time: pd.Timestamp, exit_price: float, status: str) -> None:
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.status = status
        price_diff = exit_price - trade.setup.entry_price
        if trade.setup.direction == 'short':
            price_diff = -price_diff
        gross_pnl = price_diff * trade.setup.position_size
        trade.pnl = gross_pnl

        # === Directional Prediction Evaluation ===
        y_pred = 1 if trade.setup.direction == 'long' else 0

        if trade.pnl is not None:
            if trade.setup.direction == 'long':
                y_true = 1 if trade.pnl > 0 else 0
            else:
                y_true = 1 if trade.pnl <= 0 else 0

        session = trade.session
        self.rolling_metrics[session].update(y_true, y_pred)

        # Save the latest metrics to the trade (for logging/export)
        metrics = self.rolling_metrics[session].latest()
        if hasattr(trade, "evaluation"):
            trade.evaluation.update(metrics)
        else:
            trade.evaluation = metrics

        trade.y_true = y_true
        trade.y_pred = y_pred

        self.session_capital[trade.session] += trade.pnl
        if hasattr(self.bet_sizing, "update_with_trade_result"):
            try:
                self.bet_sizing.update_with_trade_result(
                    pnl=trade.pnl,
                    risk_amount=trade.setup.risk_amount,
                    session=trade.session
                )
            except TypeError:
                # For older bet sizings that don't accept `session`
                self.bet_sizing.update_with_trade_result(
                    pnl=trade.pnl,
                    risk_amount=trade.setup.risk_amount
                )

    def get_trade_data(self) -> pd.DataFrame:
        all_trades = []
        for session_trades in self.trades.values():
            all_trades.extend([{
                'asset': self.asset.value,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.setup.entry_price,
                'exit_price': t.exit_price,
                'direction': t.setup.direction,
                'session': t.session,
                'attempt': t.setup.attempt,
                'status': t.status,
                'position_size': t.setup.position_size,
                'risk_amount': t.setup.risk_amount,
                'pnl': t.pnl,
                'return_pct': t.return_pct,
                'holding_time': t.holding_time,
                'ref_close': t.setup.ref_close,
                'date': t.entry_time.date(),
                'day_of_week': t.entry_time.day_name(),
                'equity_at_entry': t.equity_at_entry,
                'duration_minutes': (t.exit_time - t.entry_time).total_seconds() / 60 if t.exit_time else None,
                'atr_14': t.setup.atr_14,
                'ma_14': t.setup.ma_14,
                'min_price_30': t.setup.min_price_30,
                'max_price_30': t.setup.max_price_30,
                'y_true': getattr(t, 'y_true', None),
                'y_pred': getattr(t, 'y_pred', None),
                'rolling_accuracy': getattr(t, 'evaluation', {}).get('rolling_accuracy', None),
                'rolling_f1': getattr(t, 'evaluation', {}).get('rolling_f1', None),
                'rolling_precision': getattr(t, 'evaluation', {}).get('rolling_precision', None),
                'rolling_recall': getattr(t, 'evaluation', {}).get('rolling_recall', None),
                'n_total_seen': getattr(t, 'evaluation', {}).get('n_total_seen', None),
                'n_window_obs': getattr(t, 'evaluation', {}).get('n_window_obs', None),
                'session_code': self.SESSION_MAP.get(t.session, -1),
                'regime': self.data.at[t.entry_time, 'regime'] if t.entry_time in self.data.index else None,
                'regime_label': self.data.at[t.entry_time, 'regime_label'] if t.entry_time in self.data.index else None,
                'daily_return': self.data.at[t.entry_time, 'daily_return'] if t.entry_time in self.data.index else None,
                'daily_volatility': self.data.at[t.entry_time, 'daily_volatility'] if t.entry_time in self.data.index else None,
                't10yie': self.data.at[t.entry_time, 't10yie'] if t.entry_time in self.data.index else None,
                'vix_close': self.data.at[t.entry_time, 'vix_close'] if t.entry_time in self.data.index else None,


            } for t in session_trades])
            

        return pd.DataFrame(all_trades)

    def _get_session_equity(self, session: str, current_price: float) -> float:
        cash = self.session_capital[session]
        unrealized_pnl = 0.0

        for trade in self.trades[session]:
            if trade.status != 'open':
                continue

            direction = 1 if trade.setup.direction == 'long' else -1
            entry = trade.setup.entry_price
            size = trade.setup.position_size
            unrealized_pnl += (current_price - entry) * size * direction

        return cash + unrealized_pnl

    def _get_session_available_cash(self, session: str) -> float:
        cash = self.session_capital[session]
        allocated_capital = 0.0

        for trade in self.trades[session]:
            if trade.status == 'open':
                allocated_capital += trade.setup.position_size * trade.setup.entry_price

        return cash - allocated_capital



def get_bet_sizing(method: BetSizingMethod, past_returns: pd.Series = None) -> BetSizingStrategy:
    if method == BetSizingMethod.KELLY:
        return KellyBetSizing(window_size=100, fallback_fraction=0.02)
    elif method == BetSizingMethod.FIXED:
        return FixedFractionalBetSizing(investment_fraction=0.2)
    elif method == BetSizingMethod.FIXED_AMOUNT:
        return FixedBetSize(fixed_trade_size=20000)
    elif method == BetSizingMethod.PERCENT_VOLATILITY:
        return PercentVolatilityBetSizing(risk_fraction=0.01)
    elif method == BetSizingMethod.OPTIMAL_F:
        return OptimalF(min_trades=20, default_fraction=0.01)
    else:
        raise ValueError(f"Unsupported bet sizing method: {method}")



class RollingMetrics:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.y_true = deque(maxlen=window_size)
        self.y_pred = deque(maxlen=window_size)
        self.rolling_accuracy = []
        self.rolling_f1 = []
        self.rolling_precision = []
        self.rolling_recall = []
        self.total_seen = 0

    def update(self, y_true_val, y_pred_val):
        self.y_true.append(y_true_val)
        self.y_pred.append(y_pred_val)
        self.total_seen += 1


        if len(self.y_pred) == self.window_size:
            print(f"\n📊 Rolling Window Debug (last {self.window_size} trades):")
            print(f"  y_true counts: {np.bincount(self.y_true)}")
            print(f"  y_pred counts: {np.bincount(self.y_pred)}")
        
        if len(self.y_true) > 0:
            self.rolling_accuracy.append(accuracy_score(self.y_true, self.y_pred))
            self.rolling_precision.append(precision_score(self.y_true, self.y_pred, zero_division=0))
            self.rolling_recall.append(recall_score(self.y_true, self.y_pred, zero_division=0))
            self.rolling_f1.append(f1_score(self.y_true, self.y_pred, zero_division=0))
        else:
            self.rolling_accuracy.append(None)
            self.rolling_precision.append(None)
            self.rolling_recall.append(None)
            self.rolling_f1.append(None)

    def latest(self):
        def safe_get(lst):
            return lst[-1] if lst else None

        return {
            "rolling_accuracy": safe_get(self.rolling_accuracy),
            "rolling_f1": safe_get(self.rolling_f1),
            "rolling_precision": safe_get(self.rolling_precision),
            "rolling_recall": safe_get(self.rolling_recall),
            "n_total_seen": self.total_seen,
            "n_window_obs": len(self.y_true) 
        }
