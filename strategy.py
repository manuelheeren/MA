"""
Strategy Implementation
    Session-specific overnight momentum strategy 
    Each session compares its opening price to its previous session close
    Fixed SL/TP based on tick value (350 ticks ≈ $17.5)
    Re-entry mechanism with up to 3 attempts
    Session-based trade management
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from datetime import time
import logging
from collections import defaultdict

@dataclass(frozen=True)
class SessionTime:
    """Immutable session time configuration with closing times"""
    name: str
    start: time  # Session start time
    end: time    # Session end time
    close: time  # Time to use as session close (for next day comparison)

@dataclass
class TradeSetup:
    """Immutable trade setup parameters"""
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    attempt: int
    ref_close: float  # Added to store reference closing price

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
    # Market session definitions with closing times
    SESSIONS = [
        SessionTime('asian', time(0, 0), time(8, 0), time(7, 59)),
        SessionTime('london', time(8, 0), time(16, 0), time(15, 59)),
        SessionTime('us', time(13, 0), time(21, 0), time(20, 59))
    ]
    
    # Strategy constants
    TICK_VALUE = 17.5  # 350 ticks in dollars for Gold
    MAX_ATTEMPTS = 3
    
    def __init__(self, data: pd.DataFrame, risk_percent: float = 0.01):
        """Initialize strategy with price data and risk parameters"""
        self.data = data
        self.risk_percent = risk_percent
        self.trades: List[Trade] = []

    def _get_previous_session_close(self, current_time: pd.Timestamp, session: SessionTime) -> Optional[float]:
        """
        Get the closing price from the previous session
        
        Parameters:
        -----------
        current_time : pd.Timestamp
            Current time for trade entry consideration
        session : SessionTime
            Session configuration including closing time
            
        Returns:
        --------
        Optional[float]
            Previous session's closing price or None if not found
        """
        # Get previous trading day
        prev_day = current_time.normalize() - pd.Timedelta(days=1)
        
        # Find the last price at session close time
        prev_close_time = prev_day.replace(
            hour=session.close.hour,
            minute=session.close.minute
        )
        
        # Get the closest price data before or at closing time
        prev_day_data = self.data[self.data.index <= prev_close_time]
        if not prev_day_data.empty:
            return prev_day_data.iloc[-1]['close']
        return None

    def _create_trade_setup(self, price: float, direction: str, attempt: int, ref_close: float) -> TradeSetup:
        """Create trade setup with position sizing"""
        r_multiple = attempt if attempt > 1 else 1
        sl_distance = self.TICK_VALUE
        
        if direction == 'long':
            stop_loss = price - sl_distance
            take_profit = price + (sl_distance * r_multiple)
        else:
            stop_loss = price + sl_distance
            take_profit = price - (sl_distance * r_multiple)
            
        return TradeSetup(direction, price, stop_loss, take_profit, attempt, ref_close)

    def _should_enter_trade(self, current_price: float, prev_close: float) -> Optional[str]:
        """Determine trade direction based on momentum"""
        if current_price > prev_close:
            return 'long'
        elif current_price < prev_close:
            return 'short'
        return None

    def generate_signals(self) -> None:
        """Generate trading signals for all sessions with proper weekend handling"""
        for date in self.data['date'].unique():
            # Skip weekends
            if pd.Timestamp(date).weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                continue
                
            for session in self.SESSIONS:
                # Create session start timestamp
                session_start = pd.Timestamp(f"{date} {session.start}", tz='UTC')
                if session_start not in self.data.index:
                    continue
                
                # Get previous session's closing price
                prev_close = self._get_previous_session_close(session_start, session)
                if prev_close is None:
                    continue  # Skip if no previous close available
                
                # Get opening price for current session
                session_open = self.data.loc[session_start, 'close']
                
                # Check for trade setup
                direction = self._should_enter_trade(session_open, prev_close)
                if direction:
                    setup = self._create_trade_setup(session_open, direction, attempt=1, ref_close=prev_close)
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
        """Simulate trades with proper re-entry handling"""
        # Process trades chronologically
        processed_trades = []
        trades_to_process = self.trades.copy()  # Start with initial trades
        
        while trades_to_process:
            trade = trades_to_process.pop(0)  # Get next trade to process
            
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
                            trade.setup.attempt + 1,
                            trade.setup.ref_close  # Keep original reference close
                        )
                        new_trade = Trade(timestamp, setup, trade.session)
                        trades_to_process.append(new_trade)  # Add to processing queue
                    break
            
            # Close at session end if neither TP nor SL was hit
            if not trade_closed:
                last_price = session_prices.iloc[-1]['close'] if not session_prices.empty else trade.setup.entry_price
                self._close_trade(trade, session_end, last_price, 'session_close')
            
            processed_trades.append(trade)
        
        # Update trades list with all processed trades
        self.trades = processed_trades

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
            'r_multiple': t.r_multiple,
            'ref_close': t.setup.ref_close  # Added to output for analysis
        } for t in self.trades])