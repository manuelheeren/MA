from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import time
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Asset(Enum):
    """Trading assets"""
    XAUUSD = "XAUUSD"
    BTCUSD = "BTCUSD"
    SPYUSD = "SPYUSD"

@dataclass(frozen=True)
class SessionTime:
    """Session time configuration"""
    name: str
    start: time
    end: time
    close: time

@dataclass
class TradeSetup:
    """Trade setup configuration"""
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
    """Trade execution and tracking"""
    entry_time: pd.Timestamp
    setup: TradeSetup
    session: str
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    status: str = 'open'
    pnl: Optional[float] = None

    @property
    def holding_time(self) -> Optional[pd.Timedelta]:
        return self.exit_time - self.entry_time if self.exit_time else None

    @property
    def return_pct(self) -> Optional[float]:
        return (self.pnl / self.setup.risk_amount * 0.01) if self.pnl is not None else None

class TradingStrategy:
    # Constants
    SESSIONS = [
        SessionTime('asian', time(0, 0), time(8, 0), time(7, 59)),
        SessionTime('london', time(8, 0), time(16, 0), time(15, 59)),
        SessionTime('us', time(13, 0), time(21, 0), time(20, 59))
    ]
    
    MAX_ATTEMPTS = 3
    BASE_RISK_PCT = 0.005
    RISK_PCT = 0.01
    INITIAL_CAPITAL = 100000

    def __init__(self, data: pd.DataFrame, asset: str):
        self.data = data
        self.asset = Asset(asset)
        self.session_capital = {s.name: self.INITIAL_CAPITAL for s in self.SESSIONS}
        self.trades = {s.name: [] for s in self.SESSIONS}
        self.data['ATR'] = self._calculate_atr(3)
        
    def _calculate_atr(self, window: int) -> pd.Series:
        """Calculate the Average True Range (ATR) for the given window."""
        high_low = self.data['high'] - self.data['low']
        high_close = abs(self.data['high'] - self.data['close'].shift(1))
        low_close = abs(self.data['low'] - self.data['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()

    def _calculate_trade_levels(self, price: float, direction: str, attempt: int, atr: float) -> tuple:
        """Calculate stop loss and take profit levels using ATR."""
        sl_multiplier = 1.0
        tp_multiplier = 1.5 + (attempt - 1) * 0.5

        if direction == 'long':
            stop_loss = price - sl_multiplier * atr
            take_profit = price + tp_multiplier * atr
        else:
            stop_loss = price + sl_multiplier * atr
            take_profit = price - tp_multiplier * atr
        
        return stop_loss, take_profit

    def _create_trade_setup(self, price: float, direction: str, attempt: int, 
                            ref_close: float, session: str) -> TradeSetup:
        """Create a new trade setup with position sizing using current session capital."""
        atr = self.data.loc[self.data['close'] == price, 'ATR'].iloc[0]
        stop_loss, take_profit = self._calculate_trade_levels(price, direction, attempt, atr)
        
        # Calculate position size using CURRENT session capital
        current_capital = self.session_capital[session]
        position_size = (current_capital * self.RISK_PCT) / abs(price - stop_loss)
        risk_amount = abs(price - stop_loss) * position_size
        
        # Add validation
        if not (np.isfinite(position_size) and np.isfinite(risk_amount)):
            position_size = (current_capital * self.RISK_PCT) / (price * 0.01)
            risk_amount = current_capital * self.RISK_PCT
        
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
        """Generate initial trade signals for all sessions"""
        self.trade_signals = {s.name: [] for s in self.SESSIONS}
        
        for date in sorted(self.data['date'].unique()):
            if pd.Timestamp(date).weekday() >= 5:  # Skip weekends
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
        """Simulate trades with re-entry handling"""
        for session_name, signals in self.trade_signals.items():
            self._process_session_signals(session_name, signals)

    def _process_session_signals(self, session_name: str, signals: List[dict]) -> None:
        """Process all trade signals for a single session"""
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
            
            trade = Trade(entry_time, setup, session_name)
            remaining_trades = []
            
            self._process_single_trade(trade, self._get_session_prices(entry_time, session_end), 
                                     session_end, remaining_trades)
            processed_trades.append(trade)
            
            while remaining_trades:
                re_trade = remaining_trades.pop(0)
                self._process_single_trade(re_trade, 
                                         self._get_session_prices(re_trade.entry_time, session_end),
                                         session_end, remaining_trades)
                processed_trades.append(re_trade)
        
        self.trades[session_name] = processed_trades

    def _process_single_trade(self, trade: Trade, prices: pd.DataFrame, 
                            session_end: pd.Timestamp, trades_to_process: List[Trade]) -> bool:
        """Process a single trade and handle re-entry if needed"""
        for timestamp, price_data in prices.iterrows():
            # Check take profit
            if self._check_take_profit(trade, price_data):
                self._close_trade(trade, timestamp, trade.setup.take_profit, 'tp_hit')
                return True
                
            # Check stop loss and handle re-entry
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
        
        # Close at session end if neither TP nor SL was hit
        last_price = prices.iloc[-1]['close'] if not prices.empty else trade.setup.entry_price
        self._close_trade(trade, session_end, last_price, 'session_close')
        return True

    def _get_previous_session_close(self, current_time: pd.Timestamp, session: SessionTime) -> Optional[float]:
        """Get the closing price from previous session"""
        prev_close_time = (current_time.normalize() - pd.Timedelta(days=1)).replace(
            hour=session.close.hour,
            minute=session.close.minute
        )
        prev_data = self.data[self.data.index <= prev_close_time]
        return prev_data.iloc[-1]['close'] if not prev_data.empty else None

    def _check_take_profit(self, trade: Trade, price_data: pd.Series) -> bool:
        """Check if take profit level is hit"""
        return ((trade.setup.direction == 'long' and price_data['high'] >= trade.setup.take_profit) or
                (trade.setup.direction == 'short' and price_data['low'] <= trade.setup.take_profit))

    def _check_stop_loss(self, trade: Trade, price_data: pd.Series) -> bool:
        """Check if stop loss level is hit"""
        return ((trade.setup.direction == 'long' and price_data['low'] <= trade.setup.stop_loss) or
                (trade.setup.direction == 'short' and price_data['high'] >= trade.setup.stop_loss))

    def _close_trade(self, trade: Trade, exit_time: pd.Timestamp, exit_price: float, 
                    status: str) -> None:
        """Close a trade and update account"""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.status = status
        
        price_diff = exit_price - trade.setup.entry_price
        if trade.setup.direction == 'short':
            price_diff = -price_diff
            
        trade.pnl = price_diff * trade.setup.position_size
        self.session_capital[trade.session] += trade.pnl

    def get_trade_data(self) -> pd.DataFrame:
        """Convert trades to DataFrame for analysis"""
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
                'duration_minutes': (t.exit_time - t.entry_time).total_seconds() / 60 if t.exit_time else None
            } for t in session_trades])
        
        return pd.DataFrame(all_trades)