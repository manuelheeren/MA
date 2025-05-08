"""Unified backtesting implementation for multi-asset strategies with session-based tracking, fee analysis and extended metrics"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import logging
from typing import Optional, Dict, Tuple
from scipy import stats

@dataclass
class ReturnMetrics:
    """Performance metrics for strategy evaluation"""
    total_pnl: float
    total_return_pct: float
    win_rate: float
    avg_win: float
    avg_loss: float
    sharpe: Optional[float]
    skewness: Optional[float]
    excess_kurtosis: Optional[float]
    max_drawdown_pct: float
    total_trades: int
    risk_amount: float

class Backtest:
    def __init__(self, strategy, output_dir: Path = None, fee: float = 0.01):
        """
        Initialize backtest with strategy and fee level
        
        Parameters:
        -----------
        strategy : TradingStrategy
            Instance of trading strategy
        output_dir : Path, optional
            Directory for saving results
        fee : float
            Trading fee as decimal (e.g., 0.01 for 1%)
        """
        self.strategy = strategy
        self.trades_df = strategy.get_trade_data()
        self.output_dir = output_dir or Path("data/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.fee = fee
        
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

    def _calculate_performance_metrics(self, returns: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate Sharpe ratio, skewness, and kurtosis from return series"""
        if len(returns) < 3:
            return None, None, None
            
        # Remove any NaN values
        returns = returns.dropna()
        
        if len(returns) < 3:
            return None, None, None
            
        # Annualize Sharpe using actual trading days
        trading_days = len(returns)
        if trading_days < 20:
            sharpe = None
        else:
            annualized_return = returns.mean() * 252  # Annualize using 252 trading days
            annualized_vol = returns.std() * np.sqrt(252)
            sharpe = annualized_return / annualized_vol if annualized_vol != 0 else None

        try:
            # Calculate skewness and excess kurtosis
            skewness = stats.skew(returns)
            excess_kurtosis = stats.kurtosis(returns, fisher=False) - 3
        except:
            skewness = None
            excess_kurtosis = None
            
        return sharpe, skewness, excess_kurtosis

    def _calculate_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown percentage"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        return abs(drawdowns.min()) * 100

    def _calculate_return_metrics(self, trades_df: pd.DataFrame) -> ReturnMetrics:
        """Calculate comprehensive return metrics"""
        if trades_df.empty:
            return ReturnMetrics(0, 0, 0, 0, 0, None, None, None, 0, 0, 0)
        
        # Calculate daily returns for Sharpe and drawdown
        daily_returns = trades_df.groupby('date')['return_pct'].sum()
        
        # Separate winning and losing trades
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        total_pnl = trades_df['pnl'].sum()
        initial_capital = self.strategy.INITIAL_CAPITAL
        
        # Calculate performance metrics
        sharpe, skewness, excess_kurtosis = self._calculate_performance_metrics(daily_returns)
        
        return ReturnMetrics(
            total_pnl=total_pnl,
            total_return_pct=(total_pnl / initial_capital) * 100,
            win_rate=len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            avg_win=winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            avg_loss=losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            sharpe=sharpe,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
            max_drawdown_pct=self._calculate_drawdown(daily_returns),
            total_trades=len(trades_df),
            risk_amount=trades_df['risk_amount'].iloc[0] if not trades_df.empty else 0
        )

    def _export_detailed_trades(self) -> None:
        """Export detailed trade information with fee level in filename"""
        fee_str = f"{self.fee*100:.1f}".replace('.', '_')  # Convert 0.01 to "1_0"
        output_path = self.output_dir / f"trades_detailed_{self.strategy.asset.value}_fee_{fee_str}.csv"
        self.trades_df.sort_values('entry_time').to_csv(output_path, index=False)
        self.logger.info(f"Exported detailed trades to {output_path}")

    def save_results_to_file(self) -> None:
        """Save backtest results to a text file"""
        fee_str = f"{self.fee*100:.1f}".replace('.', '_')
        output_path = self.output_dir / f"backtest_results_{self.strategy.asset.value}_fee_{fee_str}.txt"
        
        with open(output_path, 'w') as f:
            f.write(f"=== {self.strategy.asset.value} Backtest Results ===\n")
            f.write(f"Trading Fee: {self.fee*100:.1f}%\n\n")
            
            if self.results['period']['start'] is not None:
                f.write(f"Period: {self.results['period']['start']:%Y-%m-%d} to {self.results['period']['end']:%Y-%m-%d}\n\n")
            
            for session, data in self.results['sessions'].items():
                metrics = data['metrics']
                f.write(f"\n{session.upper()} Session Performance:\n")
                f.write(f"Initial Capital: ${self.strategy.INITIAL_CAPITAL:,.2f}\n")
                f.write(f"Final Capital: ${(self.strategy.INITIAL_CAPITAL * (1 + metrics.total_return_pct/100)):,.2f}\n")
                f.write(f"Total PnL: ${metrics.total_pnl:,.2f}\n")
                f.write(f"Return: {metrics.total_return_pct:.2f}%\n")
                
                session_trades = self.trades_df[self.trades_df['session'] == session]
                wins = len(session_trades[session_trades['pnl'] > 0])
                losses = len(session_trades[session_trades['pnl'] <= 0])
                f.write(f"Win Rate: {metrics.win_rate:.2%} ({wins}W/{losses}L)\n")
                f.write(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%\n")
                
                if metrics.sharpe is not None:
                    f.write(f"Sharpe Ratio: {metrics.sharpe:.2f}\n")
                if metrics.skewness is not None:
                    f.write(f"Skewness: {metrics.skewness:.3f}\n")
                if metrics.excess_kurtosis is not None:
                    f.write(f"Excess Kurtosis: {metrics.excess_kurtosis:.3f}\n")
                
                # Write attempt analysis
                if data['attempts']:
                    f.write("\nAttempt Analysis:\n")
                    for attempt, attempt_metrics in data['attempts'].items():
                        attempt_trades = session_trades[session_trades['attempt'] == attempt]
                        wins = len(attempt_trades[attempt_trades['pnl'] > 0])
                        losses = len(attempt_trades[attempt_trades['pnl'] <= 0])
                        
                        f.write(f"\n  Attempt {attempt}:\n")
                        f.write(f"  Trades: {attempt_metrics.total_trades} "
                               f"({attempt_metrics.total_trades/metrics.total_trades*100:.1f}% of session trades)\n")
                        f.write(f"  PnL: ${attempt_metrics.total_pnl:,.2f}\n")
                        f.write(f"  Win Rate: {attempt_metrics.win_rate:.2%} ({wins}W/{losses}L)\n")
                        f.write(f"  Average Win: ${attempt_metrics.avg_win:,.2f}\n")
                        f.write(f"  Average Loss: ${attempt_metrics.avg_loss:,.2f}\n")
                
                f.write("\n" + "="*50 + "\n")
        
        self.logger.info(f"Saved backtest results to {output_path}")

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

        # Save results to file
        self.save_results_to_file()

    def print_summary(self) -> None:
        """Print comprehensive analysis summary"""
        if not self.results:
            self.logger.warning("No results available for analysis")
            return

        print(f"\n=== {self.strategy.asset.value} Backtest Results ===")
        print(f"Trading Fee: {self.fee*100:.1f}%\n")
        
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
            
            session_trades = self.trades_df[self.trades_df['session'] == session]
            wins = len(session_trades[session_trades['pnl'] > 0])
            losses = len(session_trades[session_trades['pnl'] <= 0])
            print(f"Win Rate: {metrics.win_rate:.2%} ({wins}W/{losses}L)")
            print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
            
            if metrics.sharpe is not None:
                print(f"Sharpe Ratio: {metrics.sharpe:.2f}")
            if metrics.skewness is not None:
                print(f"Skewness: {metrics.skewness:.3f}")
            if metrics.excess_kurtosis is not None:
                print(f"Excess Kurtosis: {metrics.excess_kurtosis:.3f}")
            
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
    Main function to run backtest analysis with different fee levels
    """
    from strategy import TradingStrategy
    from pathlib import Path
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Define test parameters
    assets = ["XAUUSD"]  # "BTCUSD", "XAUUSD", "WTI"
    fees = [0.0, 0.005, 0.01]  # 0%, 0.5%, 1%
    
    # Create output directory
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for asset_name in assets:
        try:
            # Load data
            data_path = Path(f"data/processed/{asset_name}/combined_data.csv")
            if not data_path.exists():
                logger.warning(f"Data file not found for {asset_name}, skipping...")
                continue
                
            data = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
            
            # Test each fee level
            for fee in fees:
                logger.info(f"\nTesting {asset_name} with {fee*100:.1f}% fee")
                
                # Initialize strategy with fee
                strategy = TradingStrategy(data, asset_name, trading_fee=fee)
                strategy.generate_signals()
                strategy.simulate_trades()
                
                # Run backtest
                backtest = Backtest(strategy, output_dir, fee=fee)
                backtest.run_analysis()
                backtest.print_summary()

        except Exception as e:
            logger.error(f"Error testing {asset_name}: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()