"""
Strategy Implementation
    Core momentum strategy based on previous day's close
    Entry logic at session starts
    Fixed SL/TP based on tick value (350 ticks ≈ $17.5)
    Re-entry mechanism with up to 3 attempts
    Session-based trade management
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from datetime import time
import logging

@dataclass(frozen=True)
class SessionTime:
    """Immutable session time configuration"""
    name: str
    start: time
    end: time

@dataclass
class TradeSetup:
    """Immutable trade setup parameters"""
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    attempt: int

@dataclass
class Trade:
    """Trade object with minimal required fields"""
    entry_time: pd.Timestamp
    setup: TradeSetup
    session: str
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    status: str = 'open'
    pnl: Optional[float] = None

    @property
    def holding_time(self) -> Optional[pd.Timedelta]:
        """Calculate trade holding time"""
        return self.exit_time - self.entry_time if self.exit_time else None

    @property
    def r_multiple(self) -> Optional[float]:
        """Calculate R multiple based on attempt"""
        return self.pnl / 17.5 if self.pnl is not None else None

class TradingStrategy:
    # Market session definitions
    SESSIONS = [
        SessionTime('asian', time(0, 0), time(8, 0)),
        SessionTime('london', time(8, 0), time(16, 0)),
        SessionTime('us', time(13, 0), time(21, 0))
    ]
    
    # Strategy constants
    TICK_VALUE = 17.5  # 350 ticks in dollars for Gold
    MAX_ATTEMPTS = 3
    
    def __init__(self, data: pd.DataFrame, risk_percent: float = 0.01):
        """Initialize strategy with price data and risk parameters"""
        self.data = data
        self.risk_percent = risk_percent
        self.trades: List[Trade] = []

    def _create_trade_setup(self, price: float, direction: str, attempt: int) -> TradeSetup:
        """Create trade setup with position sizing"""
        r_multiple = attempt if attempt > 1 else 1
        sl_distance = self.TICK_VALUE
        
        if direction == 'long':
            stop_loss = price - sl_distance
            take_profit = price + (sl_distance * r_multiple)
        else:
            stop_loss = price + sl_distance
            take_profit = price - (sl_distance * r_multiple)
            
        return TradeSetup(direction, price, stop_loss, take_profit, attempt)

    def _should_enter_trade(self, current_price: float, prev_close: float) -> Optional[str]:
        """Determine trade direction based on momentum"""
        if current_price > prev_close:
            return 'long'
        elif current_price < prev_close:
            return 'short'
        return None

    def generate_signals(self) -> None:
        """Generate trading signals for all sessions"""
        for date in self.data['date'].unique():
            for session in self.SESSIONS:
                # Create session start timestamp
                session_start = pd.Timestamp(f"{date} {session.start}", tz='UTC')
                if session_start not in self.data.index:
                    continue
                
                # Get data at session start
                session_data = self.data.loc[session_start]
                
                # Check for trade setup
                direction = self._should_enter_trade(session_data['close'], session_data['prev_close'])
                if direction:
                    setup = self._create_trade_setup(session_data['close'], direction, attempt=1)
                    self.trades.append(Trade(session_start, setup, session.name))

    def _close_trade(self, trade: Trade, exit_time: pd.Timestamp, exit_price: float, 
                    status: str) -> None:
        """Close trade and calculate PnL"""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.status = status
        
        if status == 'tp_hit':
            trade.pnl = self.TICK_VALUE * trade.setup.attempt
        elif status == 'sl_hit':
            trade.pnl = -self.TICK_VALUE
        else:  # session_close
            multiplier = 1 if trade.setup.direction == 'long' else -1
            trade.pnl = (exit_price - trade.setup.entry_price) * multiplier

    def simulate_trades(self) -> None:
        """
        Simulate trades with re-entry logic
        
        All trades (including re-entries):
        - Can hit TP (take profit)
        - Can hit SL (stop loss)
        - Are closed at session end if neither TP nor SL is hit
        """
        for trade in self.trades:
            # Find session end time
            session_end = next(s.end for s in self.SESSIONS if s.name == trade.session)
            session_end = pd.Timestamp(f"{trade.entry_time.date()} {session_end}", tz='UTC')
            
            # Get session price data
            mask = (self.data.index > trade.entry_time) & (self.data.index <= session_end)
            session_prices = self.data[mask]
            
            trade_closed = False
            
            for timestamp, prices in session_prices.iterrows():
                # Check take profit
                if ((trade.setup.direction == 'long' and prices['high'] >= trade.setup.take_profit) or
                    (trade.setup.direction == 'short' and prices['low'] <= trade.setup.take_profit)):
                    self._close_trade(trade, timestamp, trade.setup.take_profit, 'tp_hit')
                    trade_closed = True
                    break
                    
                # Check stop loss and handle re-entry
                elif ((trade.setup.direction == 'long' and prices['low'] <= trade.setup.stop_loss) or
                    (trade.setup.direction == 'short' and prices['high'] >= trade.setup.stop_loss)):
                    self._close_trade(trade, timestamp, trade.setup.stop_loss, 'sl_hit')
                    trade_closed = True
                    
                    # Create re-entry if attempts remain
                    if trade.setup.attempt < self.MAX_ATTEMPTS:
                        setup = self._create_trade_setup(
                            trade.setup.stop_loss,
                            trade.setup.direction,
                            trade.setup.attempt + 1
                        )
                        self.trades.append(Trade(timestamp, setup, trade.session))
                    break
            
            # Close at session end if neither TP nor SL was hit
            if not trade_closed:
                last_price = session_prices.iloc[-1]['close'] if not session_prices.empty else trade.setup.entry_price
                self._close_trade(trade, session_end, last_price, 'session_close')

    def get_trade_data(self) -> pd.DataFrame:
        """Convert trades to DataFrame for analysis"""
        return pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.setup.entry_price,
            'exit_price': t.exit_price,
            'direction': t.setup.direction,
            'session': t.session,
            'attempt': t.setup.attempt,
            'status': t.status,
            'pnl': t.pnl,
            'holding_time': t.holding_time,
            'r_multiple': t.r_multiple
        } for t in self.trades])