# new_strategy.py

from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import time
import logging
from enum import Enum
from bet_sizing import KellyBetSizing, FixedFractionalBetSizing, BetSizingStrategy, FixedBetSize

# Configure logging
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

    MAX_ATTEMPTS = 3
    INITIAL_CAPITAL = 100000

    def __init__(self, data: pd.DataFrame, asset: str, bet_sizing: BetSizingStrategy, bet_sizing_method: BetSizingMethod):
        self.data = data
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

    def _create_trade_setup(self, price: float, direction: str, attempt: int, ref_close: float, session: str) -> TradeSetup:
        stop_loss, take_profit = self._calculate_trade_levels(price, direction, attempt)
        current_capital = self._get_session_equity(session, price) # Equity
        #current_capital = self.session_capital[session] #Cash
        position_size, risk_amount = self.bet_sizing.compute_position(current_capital, price, stop_loss)
        return TradeSetup(
            direction=direction,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            attempt=attempt,
            ref_close=ref_close,
            position_size=position_size,
            risk_amount=risk_amount,
            session=session
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
                    'ref_close': prev_close
                })

    def simulate_trades(self) -> None:
        for session_name, signals in self.trade_signals.items():
            self._process_session_signals(session_name, signals)

    def _process_session_signals(self, session_name: str, signals: List[dict]) -> None:
        processed_trades = []
        for signal in signals:
            entry_time = signal['entry_time']
            session = next(s for s in self.SESSIONS if s.name == session_name)
            session_end = pd.Timestamp(f"{entry_time.date()} {session.end}", tz='UTC')
            setup = self._create_trade_setup(
                price=self.data.loc[entry_time, 'close'],
                direction=signal['direction'],
                attempt=1,
                ref_close=signal['ref_close'],
                session=session_name
            )
            equity_at_entry = self._get_session_equity(session_name, self.data.loc[entry_time, 'close'])
            trade = Trade(entry_time, setup, session_name, equity_at_entry=equity_at_entry)
            remaining_trades = []
            self._process_single_trade(trade, self._get_session_prices(entry_time, session_end), session_end, remaining_trades)
            processed_trades.append(trade)
            while remaining_trades:
                re_trade = remaining_trades.pop(0)
                self._process_single_trade(re_trade, self._get_session_prices(re_trade.entry_time, session_end), session_end, remaining_trades)
                processed_trades.append(re_trade)
        self.trades[session_name] = processed_trades

    def _process_single_trade(self, trade: Trade, prices: pd.DataFrame, session_end: pd.Timestamp, trades_to_process: List[Trade]) -> bool:
        for timestamp, price_data in prices.iterrows():
            if self._check_take_profit(trade, price_data):
                self._close_trade(trade, timestamp, trade.setup.take_profit, 'tp_hit')
                return True
            if self._check_stop_loss(trade, price_data):
                self._close_trade(trade, timestamp, trade.setup.stop_loss, 'sl_hit')
                if trade.setup.attempt < self.MAX_ATTEMPTS and timestamp < session_end:
                    new_setup = self._create_trade_setup(
                        trade.setup.stop_loss,
                        trade.setup.direction,
                        trade.setup.attempt + 1,
                        trade.setup.ref_close,
                        trade.session
                    )
                    trades_to_process.append(Trade(timestamp, new_setup, trade.session))
                return True
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
        self.session_capital[trade.session] += trade.pnl
        if hasattr(self.bet_sizing, "update_with_trade_result"):
            self.bet_sizing.update_with_trade_result(trade.pnl, trade.setup.risk_amount)

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
                'duration_minutes': (t.exit_time - t.entry_time).total_seconds() / 60 if t.exit_time else None
            } for t in session_trades])
        return pd.DataFrame(all_trades)

    def _get_session_equity(self, session: str, current_price: float) -> float:
    #"""Calculate total equity (cash + unrealized PnL) for a session"""
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


def get_bet_sizing(method: BetSizingMethod, past_returns: pd.Series = None) -> BetSizingStrategy:
    if method == BetSizingMethod.KELLY:
        return KellyBetSizing()
    elif method == BetSizingMethod.FIXED:
        return FixedFractionalBetSizing(investment_fraction=0.2)
    elif method == BetSizingMethod.FIXED_AMOUNT:
        return FixedBetSize(fixed_trade_size=2000)
    else:
        raise ValueError(f"Unsupported bet sizing method: {method}")

