"""Comprehensive backtesting framework combining performance metrics and strategy analysis"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging

@dataclass
class ReturnMetrics:
    """Core return metrics"""
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    sharpe: Optional[float]
    profitable_days: int
    losing_days: int

class Backtest:
    def __init__(self, strategy, output_dir: Path = None, initial_capital: float = 100000):
        """
        Initialize backtest with strategy and output directory
        
        Parameters:
        -----------
        strategy : TradingStrategy
            Instance of the trading strategy to analyze
        output_dir : Path, optional
            Directory for saving results. Defaults to 'datasets/results/'
        initial_capital : float, optional
            Initial capital for calculating percentage returns. Defaults to 100,000
        """
        self.strategy = strategy
        self.trades_df = strategy.get_trade_data()
        self.output_dir = output_dir or Path("datasets/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.initial_capital = initial_capital
        self._prepare_data()
        self.results = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

    def _prepare_data(self):
        """Prepare data for analysis"""
        self.trades_df['date'] = pd.to_datetime(self.trades_df['entry_time']).dt.date
        self.trades_df['day_of_week'] = pd.to_datetime(self.trades_df['entry_time']).dt.day_name()
        self.trades_df['duration_minutes'] = (
            self.trades_df['exit_time'] - self.trades_df['entry_time']
        ).dt.total_seconds() / 60
        self.trades_df['outcome'] = np.where(self.trades_df['pnl'] > 0, 'win', 'loss')
        
        # Calculate percentage returns
        self.trades_df['return_pct'] = self.trades_df['pnl'] / self.initial_capital * 100

    def _export_detailed_trades(self) -> pd.DataFrame:
        """Export detailed trade information to CSV"""
        # Create detailed trades DataFrame
        detailed_trades = pd.DataFrame([{
            # Trade timing
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'holding_time_minutes': (t.exit_time - t.entry_time).total_seconds() / 60,
            'date': t.entry_time.date(),
            'day_of_week': t.entry_time.day_name(),
            
            # Session information
            'session': t.session,
            
            # Trade setup
            'direction': t.setup.direction,
            'attempt': t.setup.attempt,
            'ref_close': t.setup.ref_close,
            
            # Price levels
            'entry_price': t.setup.entry_price,
            'exit_price': t.exit_price,
            'stop_loss': t.setup.stop_loss,
            'take_profit': t.setup.take_profit,
            
            # Trade outcome
            'status': t.status,
            'pnl': t.pnl,
            'r_multiple': t.pnl / self.strategy.TICK_VALUE if t.pnl is not None else None,
            
            # Market context at entry
            'price_vs_ref': ((t.setup.entry_price - t.setup.ref_close) / t.setup.ref_close * 100 
                            if t.setup.ref_close != 0 else None),
            
            # Risk parameters
            'risk_amount': self.strategy.TICK_VALUE,
            'reward_amount': self.strategy.TICK_VALUE * t.setup.attempt,
            'risk_reward_ratio': t.setup.attempt
            
        } for t in self.strategy.trades])
        
        # Sort by date and entry time first
        detailed_trades = detailed_trades.sort_values(['date', 'entry_time'])
        
        # Add market session time ranges
        session_times = {
            s.name: f"{s.start.strftime('%H:%M')}-{s.end.strftime('%H:%M')}"
            for s in self.strategy.SESSIONS
        }
        detailed_trades['session_time_range'] = detailed_trades['session'].map(session_times)
        
        # Calculate running total
        detailed_trades['cumulative_pnl'] = detailed_trades['pnl'].cumsum()
        
        # Export to CSV
        output_path = self.output_dir / "trades_detailed.csv"
        detailed_trades.to_csv(output_path, index=False)
        self.logger.info(f"Exported detailed trade information to {output_path}")
        
        return detailed_trades

    def _calculate_trading_days(self, trades_df: Optional[pd.DataFrame] = None) -> Dict[str, int]:
        """Calculate trading days for overall and per session"""
        if trades_df is None:
            trades_df = self.trades_df
            
        trading_days = {
            'overall': trades_df['date'].nunique(),
            'asian': trades_df[trades_df['session'] == 'asian']['date'].nunique(),
            'london': trades_df[trades_df['session'] == 'london']['date'].nunique(),
            'us': trades_df[trades_df['session'] == 'us']['date'].nunique()
        }
        
        return trading_days

    def _calculate_sharpe(self, trades_df: pd.DataFrame, session: str = None) -> Optional[float]:
        """Calculate annualized Sharpe ratio"""
        if len(trades_df) < 20:
            return None
            
        # For session-specific calculation, filter trades
        if session:
            trades_df = trades_df[trades_df['session'] == session]
            
        # Calculate daily returns
        daily_returns = trades_df.groupby('date')['return_pct'].sum()
        
        if len(daily_returns) < 20 or daily_returns.std() == 0:
            return None
            
        # Get appropriate trading days for annualization
        trading_days = self._calculate_trading_days(trades_df)
        annualization_factor = (trading_days['overall'] if not session 
                              else trading_days[session])
            
        if annualization_factor < 20:
            return None
            
        # Calculate annualized Sharpe
        annualized_return = daily_returns.mean() * annualization_factor
        annualized_vol = daily_returns.std() * np.sqrt(annualization_factor)
        
        return annualized_return / annualized_vol

    def _calculate_daily_returns(self, trades_df: pd.DataFrame) -> pd.Series:
        """Calculate daily returns from trades"""
        return trades_df.groupby('date')['pnl'].sum()

    def _calculate_return_metrics(self, trades_df: pd.DataFrame, session: str = None) -> ReturnMetrics:
        """Calculate core return metrics for a set of trades"""
        daily_pnl = self._calculate_daily_returns(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]

        return ReturnMetrics(
            total_pnl=trades_df['pnl'].sum(),
            win_rate=len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            avg_win=winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            avg_loss=losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            sharpe=self._calculate_sharpe(trades_df, session),
            profitable_days=(daily_pnl > 0).sum(),
            losing_days=(daily_pnl < 0).sum()
        )

    def analyze_time_patterns(self) -> Dict:
        """Analyze performance across different time periods"""
        daily_pnl = self._calculate_daily_returns(self.trades_df)
        
        return {
            'daily_stats': {
                'best_day': {'date': daily_pnl.idxmax(), 'pnl': daily_pnl.max()},
                'worst_day': {'date': daily_pnl.idxmin(), 'pnl': daily_pnl.min()}
            },
            'day_of_week': {
                day: self._calculate_return_metrics(day_data)
                for day, day_data in self.trades_df.groupby('day_of_week')
            }
        }

    def analyze_sessions(self) -> Dict:
        """Analyze performance by trading session"""
        return {
            session: {
                'metrics': self._calculate_return_metrics(session_data, session),
                'trade_count': len(session_data),
                'avg_duration': session_data['duration_minutes'].mean(),
                'attempt_distribution': session_data['attempt'].value_counts().to_dict()
            }
            for session, session_data in self.trades_df.groupby('session')
        }

    def analyze_attempt_patterns(self) -> Dict:
        """Analyze re-entry attempt performance"""
        return {
            attempt: self._calculate_return_metrics(attempt_data)
            for attempt, attempt_data in self.trades_df.groupby('attempt')
        }

    def run_analysis(self) -> pd.DataFrame:
        """Run comprehensive analysis and automatically export trade details"""
        if self.trades_df.empty:
            raise ValueError("No trades available for analysis")

        # Export detailed trades first (useful for debugging)
        detailed_trades = self._export_detailed_trades()

        self.results = {
            'overall': self._calculate_return_metrics(self.trades_df),
            'time_patterns': self.analyze_time_patterns(),
            'sessions': self.analyze_sessions(),
            'attempts': self.analyze_attempt_patterns()
        }

        return detailed_trades  # Return for potential further analysis

    def plot_analysis(self) -> None:
        """Create comprehensive analysis plots"""
        if not self.results:
            raise ValueError("Run analysis before plotting")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. PnL Distribution
        sns.histplot(data=self.trades_df, x='pnl', bins=30, ax=ax1)
        ax1.axvline(x=0, color='r', linestyle='--')
        ax1.set_title('PnL Distribution')

        # 2. Performance by Session
        session_pnl = pd.Series({
            session: data['metrics'].total_pnl
            for session, data in self.results['sessions'].items()
        })
        session_pnl.plot(kind='bar', ax=ax2)
        ax2.set_title('Total PnL by Session')

        # 3. Daily PnL Over Time
        daily_pnl = self._calculate_daily_returns(self.trades_df)
        daily_pnl.plot(ax=ax3)
        ax3.set_title('Daily PnL Over Time')
        ax3.grid(True)

        # 4. Win Rate by Session and Attempt
        win_rates = []
        for session, session_data in self.trades_df.groupby(['session', 'attempt']):
            win_rates.append({
                'session': session[0],
                'attempt': session[1],
                'win_rate': len(session_data[session_data['pnl'] > 0]) / len(session_data)
            })
        win_rates_df = pd.DataFrame(win_rates)
        win_rates_pivot = win_rates_df.pivot(
            index='session', columns='attempt', values='win_rate'
        )
        sns.heatmap(win_rates_pivot, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax4)
        ax4.set_title('Win Rate by Session and Attempt')

        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "backtest_analysis.png"
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved analysis plots to {plot_path}")

    def print_summary(self) -> None:
        """Print comprehensive analysis summary"""
        if not self.results:
            raise ValueError("Run analysis before printing summary")

        print("\n=== Overall Performance ===")
        # Get total trades and win/loss counts
        total_trades = len(self.trades_df)
        winning_trades = len(self.trades_df[self.trades_df['pnl'] > 0])
        losing_trades = len(self.trades_df[self.trades_df['pnl'] <= 0])
        
        print(f"Total Trades: {total_trades}")
        self._print_metrics("Strategy", self.results['overall'], 
                          show_win_loss_count=(winning_trades, losing_trades))

        print("\n=== Session Analysis ===")
        for session, data in self.results['sessions'].items():
            print(f"\n{session.upper()}:")
            print(f"Number of Trades: {data['trade_count']}")
            print(f"Average Duration: {data['avg_duration']:.1f} minutes")
            self._print_metrics("Performance", data['metrics'])
        
        print("\n=== Time Analysis ===")
        daily = self.results['time_patterns']['daily_stats']
        print(f"Best Day: {daily['best_day']['date']} (${daily['best_day']['pnl']:.2f})")
        print(f"Worst Day: {daily['worst_day']['date']} (${daily['worst_day']['pnl']:.2f})")

        print("\n=== Re-entry Analysis ===")
        for attempt, metrics in self.results['attempts'].items():
            # Count trades for this attempt
            attempt_trades = len(self.trades_df[self.trades_df['attempt'] == attempt])
            print(f"\nAttempt {attempt} (Trades: {attempt_trades}):")
            self._print_metrics("Performance", metrics, show_profit_days=False)

    def _print_metrics(self, label: str, metrics: ReturnMetrics, 
                        show_win_loss_count: tuple = None,
                        show_profit_days: bool = True) -> None:
            """Helper to print formatted metrics"""
            print(f"{label}:")
            print(f"  Total PnL: ${metrics.total_pnl:.2f}")
            
            # Show win rate with optional win/loss count
            if show_win_loss_count:
                win_count, loss_count = show_win_loss_count
                print(f"  Win Rate: {metrics.win_rate:.1%} ({win_count}W/{loss_count}L)")
            else:
                print(f"  Win Rate: {metrics.win_rate:.1%}")
                
            print(f"  Avg Win: ${metrics.avg_win:.2f}")
            print(f"  Avg Loss: ${metrics.avg_loss:.2f}")
            
            if metrics.sharpe is not None:
                print(f"  Sharpe Ratio: {metrics.sharpe:.2f}")
                
            if show_profit_days:
                print(f"  Profitable/Losing Days: {metrics.profitable_days}/{metrics.losing_days}")

def main():
    """Example usage of the Backtest class"""
    from strategy import TradingStrategy
    from data_handler import DukasCopyDataHandler
    
    # Load and prepare data
    data_path = Path("datasets/raw/XAUUSD/Nov23-Nov24/XAUUSD_1M_BID.csv")
    handler = DukasCopyDataHandler()
    data = handler.load_data(data_path)
    
    # Initialize and run strategy
    strategy = TradingStrategy(data)
    strategy.generate_signals()
    strategy.simulate_trades()
    
    # Run backtest with default 100k capital
    backtest = Backtest(strategy)
    detailed_trades = backtest.run_analysis()  # This will automatically create the CSV
    backtest.plot_analysis()
    backtest.print_summary()

if __name__ == "__main__":
    main()