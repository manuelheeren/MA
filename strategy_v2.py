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
    atr: float  # Added ATR field

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
    ATR_PERIOD = 3

    def __init__(self, data: pd.DataFrame, asset: str):
        self.data = self._prepare_data(data)
        self.asset = Asset(asset)
        self.session_capital = {s.name: self.INITIAL_CAPITAL for s in self.SESSIONS}
        self.trades = {s.name: [] for s in self.SESSIONS}
        
        logger.info(f"Strategy initialized for {self.asset.value}")

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data by calculating ATR"""
        df = data.copy()
        
        # Calculate True Range
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Calculate ATR with minimum value to prevent division by zero
        df['atr'] = df['tr'].rolling(window=self.ATR_PERIOD).mean()
        df['atr'] = df['atr'].fillna(method='bfill')  # Backfill NaN values
        df['atr'] = df['atr'].fillna(df['tr'])  # Fill any remaining NaNs with TR
        
        # Ensure ATR is never zero or too small
        min_atr = df['atr'].mean() * 0.001  # Use 0.1% of mean ATR as minimum
        df['atr'] = df['atr'].clip(lower=min_atr)
        
        # Drop intermediate columns
        df.drop(['high_low', 'high_close', 'low_close', 'tr'], axis=1, inplace=True)
        
        return df

    def _get_session_prices(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
        """Get price data for a specific session period"""
        mask = (self.data.index > start_time) & (self.data.index <= end_time)
        return self.data[mask]

    def _calculate_trade_levels(self, price: float, direction: str, attempt: int, atr: float) -> tuple:
        """Calculate volatility-adjusted stop loss and take profit levels"""
        # Ensure minimum ATR value
        atr = max(atr, price * 0.0001)  # Minimum 0.01% volatility
        
        sl_pct = self.BASE_RISK_PCT * atr  # Scale base risk by ATR
        tp_pct = self.BASE_RISK_PCT * attempt * atr  # Scale take profit by ATR and attempt
        
        # Ensure minimum distance between price and levels
        min_distance = price * 0.0001  # Minimum 0.01% distance
        
        if direction == 'long':
            stop_loss = price * (1 - sl_pct)
            take_profit = price * (1 + tp_pct)
            # Ensure minimum distance
            stop_loss = min(stop_loss, price - min_distance)
        else:
            stop_loss = price * (1 + sl_pct)
            take_profit = price * (1 - tp_pct)
            # Ensure minimum distance
            stop_loss = max(stop_loss, price + min_distance)
            
        return stop_loss, take_profit

    def _create_trade_setup(self, price: float, direction: str, attempt: int, 
                          ref_close: float, session: str, atr: float) -> TradeSetup:
        """Create a new trade setup with volatility-adjusted position sizing"""
        stop_loss, take_profit = self._calculate_trade_levels(price, direction, attempt, atr)
        
        # Calculate position size with safety checks
        price_diff = abs(price - stop_loss)
        if price_diff < price * 0.0001:  # Prevent too small differences
            price_diff = price * 0.0001
            
        position_size = (self.session_capital[session] * self.RISK_PCT) / price_diff
        risk_amount = price_diff * position_size
        
        # Validate calculations
        if not (np.isfinite(position_size) and np.isfinite(risk_amount)):
            logger.warning(f"Invalid position calculation: price={price}, stop_loss={stop_loss}, "
                         f"atr={atr}, position_size={position_size}, risk_amount={risk_amount}")
            # Use fallback values
            position_size = (self.session_capital[session] * self.RISK_PCT) / (price * 0.01)
            risk_amount = self.session_capital[session] * self.RISK_PCT
        
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
            atr=atr
        )

    def generate_signals(self) -> None:
        """Generate trading signals for all sessions"""
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
                current_atr = self.data.loc[session_start, 'atr']
                direction = 'long' if current_price > prev_close else 'short'
                
                setup = self._create_trade_setup(
                    current_price, direction, attempt=1,
                    ref_close=prev_close, session=session.name,
                    atr=current_atr
                )
                self.trades[session.name].append(Trade(session_start, setup, session.name))

    def simulate_trades(self) -> None:
        """Simulate trades with re-entry handling"""
        for session_name, session_trades in self.trades.items():
            self._process_session_trades(session_name, session_trades)

    def _process_session_trades(self, session_name: str, session_trades: List[Trade]) -> None:
        """Process all trades for a single session"""
        processed_trades = []
        trades_to_process = session_trades.copy()
        
        while trades_to_process:
            trade = trades_to_process.pop(0)
            session_end = self._get_session_end(trade)
            session_prices = self._get_session_prices(trade.entry_time, session_end)
            
            if self._process_single_trade(trade, session_prices, session_end, trades_to_process):
                processed_trades.append(trade)
        
        self.trades[session_name] = processed_trades
        logger.info(f"Session {session_name} completed: ${self.session_capital[session_name]:,.2f}")

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
                    current_atr = price_data['atr']
                    new_setup = self._create_trade_setup(
                        trade.setup.stop_loss,
                        trade.setup.direction,
                        trade.setup.attempt + 1,
                        trade.setup.ref_close,
                        trade.session,
                        current_atr
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

    def _get_session_end(self, trade: Trade) -> pd.Timestamp:
        """Get session end time for a trade"""
        session = next(s for s in self.SESSIONS if s.name == trade.session)
        return pd.Timestamp(f"{trade.entry_time.date()} {session.end}", tz='UTC')

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
        
        # Validate PnL before updating capital
        if np.isfinite(trade.pnl):
            self.session_capital[trade.session] += trade.pnl
        else:
            logger.warning(f"Invalid PnL calculation: price_diff={price_diff}, "
                         f"position_size={trade.setup.position_size}, pnl={trade.pnl}")

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
                'atr': t.setup.atr,  # Added ATR to output
                'date': t.entry_time.date(),
                'day_of_week': t.entry_time.day_name(),
                'duration_minutes': (t.exit_time - t.entry_time).total_seconds() / 60 if t.exit_time else None
            } for t in session_trades])
        
        return pd.DataFrame(all_trades)