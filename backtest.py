"""Unified backtesting implementation for multi-asset strategies with session-based tracking"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import logging
from typing import Optional, Dict

@dataclass
class ReturnMetrics:
    """Performance metrics for strategy evaluation"""
    total_pnl: float
    total_return_pct: float
    win_rate: float
    avg_win: float
    avg_loss: float
    sharpe: Optional[float]
    max_drawdown_pct: float
    total_trades: int
    risk_amount: float

class Backtest:
    def __init__(self, strategy, output_dir: Path = None):
        """
        Initialize backtest with strategy
        
        Parameters:
        -----------
        strategy : TradingStrategy
            Instance of either volatility-based or fixed-percentage strategy
        output_dir : Path, optional
            Directory for saving results
        """
        self.strategy = strategy
        self.trades_df = strategy.get_trade_data()
        self.output_dir = output_dir or Path("datasets/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

    def _calculate_sharpe(self, returns: pd.Series) -> Optional[float]:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 20:
            return None
        
        trading_days = len(returns)
        if trading_days < 20:
            return None
        
        annualized_return = returns.mean() * trading_days
        annualized_vol = returns.std() * np.sqrt(trading_days)
        
        return None if annualized_vol == 0 else annualized_return / annualized_vol

    def _calculate_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown percentage"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        return abs(drawdowns.min()) * 100

    def _calculate_return_metrics(self, trades_df: pd.DataFrame) -> ReturnMetrics:
        """Calculate comprehensive return metrics"""
        if trades_df.empty:
            return ReturnMetrics(0, 0, 0, 0, 0, None, 0, 0, 0)
        
        # Calculate daily returns for Sharpe and drawdown
        daily_returns = trades_df.groupby('date')['return_pct'].sum()
        
        # Separate winning and losing trades
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        total_pnl = trades_df['pnl'].sum()
        initial_capital = self.strategy.INITIAL_CAPITAL

        return ReturnMetrics(
            total_pnl=total_pnl,
            total_return_pct=(total_pnl / initial_capital) * 100,
            win_rate=len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            avg_win=winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            avg_loss=losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            sharpe=self._calculate_sharpe(daily_returns),
            max_drawdown_pct=self._calculate_drawdown(daily_returns),
            total_trades=len(trades_df),
            risk_amount=trades_df['risk_amount'].iloc[0] if not trades_df.empty else 0
        )

    def run_analysis(self) -> None:
        """Run comprehensive backtest analysis per session"""
        if self.trades_df is None or self.trades_df.empty:
            self.logger.warning(f"No trades generated for {self.strategy.asset.value}")
            self.results = {
                'asset': self.strategy.asset.value,
                'sessions': {},
                'period': {'start': None, 'end': None}
            }
            return

        # Export detailed trades
        self._export_detailed_trades()

        # Initialize results structure
        self.results = {
            'asset': self.strategy.asset.value,
            'sessions': {},
            'period': {
                'start': self.trades_df['entry_time'].min(),
                'end': self.trades_df['entry_time'].max()
            }
        }

        # Analyze each session independently
        for session in ['asian', 'london', 'us']:
            session_trades = self.trades_df[self.trades_df['session'] == session]
            
            if session_trades.empty:
                self.logger.debug(f"No trades for {session} session in {self.strategy.asset.value}")
                continue
            
            # Get session metrics
            session_metrics = self._calculate_return_metrics(session_trades)
            
            # Analyze attempts within session
            attempt_metrics = {}
            for attempt in sorted(session_trades['attempt'].unique()):
                attempt_trades = session_trades[session_trades['attempt'] == attempt]
                metrics = self._calculate_return_metrics(attempt_trades)
                attempt_metrics[attempt] = metrics
            
            # Store session results
            self.results['sessions'][session] = {
                'metrics': session_metrics,
                'attempts': attempt_metrics
            }

    def _export_detailed_trades(self) -> None:
        """Export detailed trade information sorted by entry time"""
        output_path = self.output_dir / f"trades_detailed_{self.strategy.asset.value}.csv"
        self.trades_df.sort_values('entry_time').to_csv(output_path, index=False)
        self.logger.info(f"Exported detailed trades to {output_path}")

    def print_summary(self) -> None:
        """Print comprehensive analysis summary"""
        if not self.results:
            self.logger.warning("No results available for analysis")
            return

        print(f"\n=== {self.strategy.asset.value} Backtest Results ===")
        
        if not self.results['sessions']:
            print("No trades were generated during the test period.")
            return
            
        if self.results['period']['start'] is not None:
            print(f"Period: {self.results['period']['start']:%Y-%m-%d} to {self.results['period']['end']:%Y-%m-%d}")
        
        for session, data in self.results['sessions'].items():
            metrics = data['metrics']
            print(f"\n{session.upper()} Session Performance:")
            print(f"Initial Capital: ${self.strategy.INITIAL_CAPITAL:,.2f}")
            print(f"Final Capital: ${(self.strategy.INITIAL_CAPITAL * (1 + metrics.total_return_pct/100)):,.2f}")
            print(f"Total PnL: ${metrics.total_pnl:,.2f}")
            print(f"Return: {metrics.total_return_pct:.2f}%")
            
            # Calculate session-specific win/loss counts
            session_trades = self.trades_df[self.trades_df['session'] == session]
            wins = len(session_trades[session_trades['pnl'] > 0])
            losses = len(session_trades[session_trades['pnl'] <= 0])
            print(f"Win Rate: {metrics.win_rate:.2%} ({wins}W/{losses}L)")
            print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
            if metrics.sharpe is not None:
                print(f"Sharpe Ratio: {metrics.sharpe:.2f}")
            
            # Print attempt analysis
            if data['attempts']:
                print("\nAttempt Analysis:")
                for attempt, attempt_metrics in data['attempts'].items():
                    attempt_trades = session_trades[session_trades['attempt'] == attempt]
                    wins = len(attempt_trades[attempt_trades['pnl'] > 0])
                    losses = len(attempt_trades[attempt_trades['pnl'] <= 0])
                    
                    print(f"\n  Attempt {attempt}:")
                    print(f"  Trades: {attempt_metrics.total_trades} "
                        f"({attempt_metrics.total_trades/metrics.total_trades*100:.1f}% of session trades)")
                    print(f"  PnL: ${attempt_metrics.total_pnl:,.2f}")
                    print(f"  Win Rate: {attempt_metrics.win_rate:.2%} ({wins}W/{losses}L)")
                    print(f"  Average Win: ${attempt_metrics.avg_win:,.2f}")
                    print(f"  Average Loss: ${attempt_metrics.avg_loss:,.2f}")

def main():
    """
    Main function to run backtest analysis on both strategy versions.
    Tests each strategy with both XAUUSD and BTCUSD.
    """
    from strategy import TradingStrategy as SimpleStrategy
    from strategy_v2 import TradingStrategy as VolatilityStrategy
    from pathlib import Path
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Define assets to test
    assets = ["SPYUSD"] # "XAUUSD", "BTCUSD",
    
    # Test both strategy versions
    for strategy_version, Strategy in [
        ("Simple Fixed", SimpleStrategy),
        # ("Volatility-Based", VolatilityStrategy),
    ]:
        print(f"\n{'='*50}")
        print(f"Testing {strategy_version} Strategy")
        print(f"{'='*50}")
        
        # Test each asset
        for asset_name in assets:
            try:
                # Load data
                data_path = Path(f"datasets/processed/{asset_name}/combined_data.csv")
                if not data_path.exists():
                    logger.warning(f"Data file not found for {asset_name}, skipping...")
                    continue
                    
                # Read data and get date range
                data = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
                start_date = data.index.min().strftime('%Y-%m-%d')
                end_date = data.index.max().strftime('%Y-%m-%d')
                
                logger.info(f"Processing {asset_name} ({start_date} to {end_date}) with {strategy_version} strategy")
                
                # Initialize and run strategy
                strategy = Strategy(data, asset_name)
                strategy.generate_signals()
                strategy.simulate_trades()
                
                # Run backtest
                backtest = Backtest(strategy)
                backtest.run_analysis()
                backtest.print_summary()
                
            except Exception as e:
                logger.error(f"Error testing {asset_name}: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()