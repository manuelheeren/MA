"""Market regime analysis module for evaluating strategy performance across different market conditions"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import logging
from typing import Dict, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RegimeMetrics:
    """Performance metrics for a specific regime"""
    n_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    sharpe: Optional[float]

class MarketAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.sessions = ['asian', 'london', 'us']
        
        # Technical parameters
        self.ema_short = 20
        self.ema_long = 200
        self.atr_period = 14
        
        # Prepare daily data
        self._prepare_daily_data()
        
    def _prepare_daily_data(self) -> None:
        """Prepare daily data with technical indicators"""
        # Create daily OHLCV
        self.daily_data = self.data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Calculate EMAs for trend
        self.daily_data['ema_short'] = self.daily_data['close'].ewm(span=self.ema_short).mean()
        self.daily_data['ema_long'] = self.daily_data['close'].ewm(span=self.ema_long).mean()
        
        # Calculate ATR for volatility
        tr = pd.DataFrame({
            'hl': self.daily_data['high'] - self.daily_data['low'],
            'hc': abs(self.daily_data['high'] - self.daily_data['close'].shift(1)),
            'lc': abs(self.daily_data['low'] - self.daily_data['close'].shift(1))
        })
        self.daily_data['atr'] = tr.max(axis=1).rolling(window=self.atr_period).mean()
        
    def _classify_regimes(self) -> pd.DataFrame:
        """Classify market into trend and volatility regimes"""
        # Trend classification
        price_above_long = self.daily_data['close'] > self.daily_data['ema_long']
        short_above_long = self.daily_data['ema_short'] > self.daily_data['ema_long']
        
        trends = pd.Series(index=self.daily_data.index, dtype='object')
        trends[price_above_long & short_above_long] = 'uptrend'
        trends[~price_above_long & ~short_above_long] = 'downtrend'
        trends[trends.isna()] = 'sideways'
        
        # Volatility classification
        vol_terciles = self.daily_data['atr'].quantile([0.33, 0.67])
        volatility = pd.Series(index=self.daily_data.index, dtype='object')
        volatility[self.daily_data['atr'] <= vol_terciles[0.33]] = 'low'
        volatility[self.daily_data['atr'] >= vol_terciles[0.67]] = 'high'
        volatility[volatility.isna()] = 'medium'
        
        return pd.DataFrame({
            'trend': trends,
            'volatility': volatility
        })

    def _calculate_metrics(self, trades: pd.DataFrame) -> RegimeMetrics:
        """Calculate performance metrics for a set of trades"""
        if len(trades) == 0:
            return RegimeMetrics(0, 0.0, 0.0, 0.0, None)
        
        n_trades = len(trades)
        win_rate = len(trades[trades['pnl'] > 0]) / n_trades
        total_pnl = trades['pnl'].sum()
        avg_pnl = trades['pnl'].mean()
        
        # Calculate Sharpe if enough trades
        if n_trades >= 20:
            daily_returns = trades.groupby('date')['pnl'].sum()
            sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else None
        else:
            sharpe = None
        
        return RegimeMetrics(n_trades, win_rate, total_pnl, avg_pnl, sharpe)

    def analyze_regime_performance(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze strategy performance in different market regimes"""
        if len(trades_df) == 0:
            logger.warning("No trades to analyze")
            return {}
        
        # Get market regimes
        regimes = self._classify_regimes()
        
        # Convert trade times to dates for regime matching
        trades_df = trades_df.copy()
        trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date
        regimes['date'] = regimes.index.date
        
        # Merge trades with regimes
        trades_with_regimes = pd.merge(trades_df, regimes, on='date', how='left')
        
        results = {
            'overall': {
                'trend': {},
                'volatility': {}
            },
            'sessions': {
                session: {
                    'trend': {},
                    'volatility': {}
                } for session in self.sessions
            }
        }
        
        # Overall analysis
        # Trend regimes
        for trend in ['uptrend', 'downtrend', 'sideways']:
            trend_trades = trades_with_regimes[trades_with_regimes['trend'] == trend]
            results['overall']['trend'][trend] = self._calculate_metrics(trend_trades)
        
        # Volatility regimes
        for vol in ['high', 'medium', 'low']:
            vol_trades = trades_with_regimes[trades_with_regimes['volatility'] == vol]
            results['overall']['volatility'][vol] = self._calculate_metrics(vol_trades)
        
        # Session-specific analysis
        for session in self.sessions:
            session_trades = trades_with_regimes[trades_with_regimes['session'] == session]
            
            # Trend regimes per session
            for trend in ['uptrend', 'downtrend', 'sideways']:
                trend_trades = session_trades[session_trades['trend'] == trend]
                results['sessions'][session]['trend'][trend] = self._calculate_metrics(trend_trades)
            
            # Volatility regimes per session
            for vol in ['high', 'medium', 'low']:
                vol_trades = session_trades[session_trades['volatility'] == vol]
                results['sessions'][session]['volatility'][vol] = self._calculate_metrics(vol_trades)
        
        return results

def analyze_assets(assets: List[str] = None, sample_days: int = 30) -> None:
    """Run analysis on specified assets using recent data sample"""
    if assets is None:
        assets = ["XAUUSD", "BTCUSD", "WTI"] # "BTCUSD"
    
    for asset_name in assets:
        try:
            data_path = Path(f"data/processed/{asset_name}/combined_data.csv")
            if not data_path.exists():
                logger.warning(f"No processed data found for {asset_name}, skipping...")
                continue
            
            # Read data and get recent subset
            logger.info(f"\nAnalyzing {asset_name}...")
            data = pd.read_csv(data_path, parse_dates=['timestamp'])
            data.set_index('timestamp', inplace=True)
            
            # Get date range for analysis
            start_date = data.index.min()
            end_date = data.index.max()
            
            # Apply sample days filter if specified
            if sample_days is not None:
                start_date = end_date - pd.Timedelta(days=sample_days)
                data = data[data.index >= start_date].copy()
            
            logger.info(f"Period: {start_date.date()} to {end_date.date()}")
            
            # Run strategy
            from strategy import TradingStrategy
            strategy = TradingStrategy(data, asset_name)
            strategy.generate_signals()
            strategy.simulate_trades()
            trades_df = strategy.get_trade_data()
            
            if len(trades_df) == 0:
                logger.warning("No trades generated in the sample period")
                continue
            
            # Analyze performance across regimes
            analyzer = MarketAnalysis(data)
            results = analyzer.analyze_regime_performance(trades_df)
            
            print_regime_analysis(asset_name, results)
            
        except Exception as e:
            logger.error(f"Error analyzing {asset_name}: {str(e)}", exc_info=True)

def print_regime_analysis(asset_name: str, results: Dict) -> None:
    """Print regime analysis results"""
    print(f"\n{'='*70}")
    print(f"Regime Analysis for {asset_name}")
    print(f"{'='*70}")
    
    # Print overall trend regime performance
    print("\nOVERALL TREND REGIME PERFORMANCE:")
    print(f"{'Regime':<12} {'Trades':<8} {'Win Rate':<10} {'Total P&L':<12} {'Avg P&L':<10} {'Sharpe':<8}")
    print("-" * 70)
    
    for regime, metrics in results['overall']['trend'].items():
        if metrics.n_trades > 0:
            print(f"{regime:<12} {metrics.n_trades:<8d} {metrics.win_rate:>8.1%} "
                  f"${metrics.total_pnl:>10,.2f} ${metrics.avg_pnl:>8,.2f} "
                  f"{metrics.sharpe:>6.2f}" if metrics.sharpe else "   N/A")
    
    # Print overall volatility regime performance
    print("\nOVERALL VOLATILITY REGIME PERFORMANCE:")
    print(f"{'Regime':<12} {'Trades':<8} {'Win Rate':<10} {'Total P&L':<12} {'Avg P&L':<10} {'Sharpe':<8}")
    print("-" * 70)
    
    for regime, metrics in results['overall']['volatility'].items():
        if metrics.n_trades > 0:
            print(f"{regime:<12} {metrics.n_trades:<8d} {metrics.win_rate:>8.1%} "
                  f"${metrics.total_pnl:>10,.2f} ${metrics.avg_pnl:>8,.2f} "
                  f"{metrics.sharpe:>6.2f}" if metrics.sharpe else "   N/A")
    
    # Print session-specific performance
    for session in results['sessions']:
        print(f"\n{session.upper()} SESSION - TREND REGIMES:")
        print(f"{'Regime':<12} {'Trades':<8} {'Win Rate':<10} {'Total P&L':<12} {'Avg P&L':<10} {'Sharpe':<8}")
        print("-" * 70)
        
        for regime, metrics in results['sessions'][session]['trend'].items():
            if metrics.n_trades > 0:
                print(f"{regime:<12} {metrics.n_trades:<8d} {metrics.win_rate:>8.1%} "
                      f"${metrics.total_pnl:>10,.2f} ${metrics.avg_pnl:>8,.2f} "
                      f"{metrics.sharpe:>6.2f}" if metrics.sharpe else "   N/A")
        
        print(f"\n{session.upper()} SESSION - VOLATILITY REGIMES:")
        print(f"{'Regime':<12} {'Trades':<8} {'Win Rate':<10} {'Total P&L':<12} {'Avg P&L':<10} {'Sharpe':<8}")
        print("-" * 70)
        
        for regime, metrics in results['sessions'][session]['volatility'].items():
            if metrics.n_trades > 0:
                print(f"{regime:<12} {metrics.n_trades:<8d} {metrics.win_rate:>8.1%} "
                      f"${metrics.total_pnl:>10,.2f} ${metrics.avg_pnl:>8,.2f} "
                      f"{metrics.sharpe:>6.2f}" if metrics.sharpe else "   N/A")

if __name__ == "__main__":
    # Analyze full datasets for both assets
    analyze_assets(assets=["XAUUSD", "BTCUSD", "WTI"], sample_days=None)