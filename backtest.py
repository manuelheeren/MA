"""Comprehensive backtesting framework combining performance metrics and strategy analysis"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
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
    def __init__(self, strategy):
        self.strategy = strategy
        self.trades_df = strategy.get_trade_data()
        self._prepare_data()
        self.results = {}

    def _prepare_data(self):
        """Prepare data for analysis"""
        self.trades_df['date'] = pd.to_datetime(self.trades_df['entry_time']).dt.date
        self.trades_df['day_of_week'] = pd.to_datetime(self.trades_df['entry_time']).dt.day_name()
        self.trades_df['duration_minutes'] = (
            self.trades_df['exit_time'] - self.trades_df['entry_time']
        ).dt.total_seconds() / 60
        self.trades_df['outcome'] = np.where(self.trades_df['pnl'] > 0, 'win', 'loss')

    def _calculate_daily_returns(self, trades_df: pd.DataFrame) -> pd.Series:
        """Calculate daily returns from trades"""
        return trades_df.groupby('date')['pnl'].sum()

    def _calculate_sharpe(self, daily_returns: pd.Series) -> Optional[float]:
        """Calculate annualized Sharpe ratio"""
        if len(daily_returns) < 20:
            return None
        annualized_return = daily_returns.mean() * 252
        annualized_vol = daily_returns.std() * np.sqrt(252)
        return None if annualized_vol == 0 else annualized_return / annualized_vol

    def _calculate_return_metrics(self, trades_df: pd.DataFrame) -> ReturnMetrics:
        """Calculate core return metrics for a set of trades"""
        daily_returns = self._calculate_daily_returns(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]

        return ReturnMetrics(
            total_pnl=trades_df['pnl'].sum(),
            win_rate=len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            avg_win=winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            avg_loss=losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            sharpe=self._calculate_sharpe(daily_returns),
            profitable_days=(daily_returns > 0).sum(),
            losing_days=(daily_returns < 0).sum()
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
                'metrics': self._calculate_return_metrics(session_data),
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

    def run_analysis(self):
        """Run comprehensive analysis"""
        if self.trades_df.empty:
            raise ValueError("No trades available for analysis")

        self.results = {
            'overall': self._calculate_return_metrics(self.trades_df),
            'time_patterns': self.analyze_time_patterns(),
            'sessions': self.analyze_sessions(),
            'attempts': self.analyze_attempt_patterns()
        }

    def plot_analysis(self, save_path: Optional[Path] = None):
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
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def print_summary(self):
        """Print comprehensive analysis summary with calculations"""
        if not self.results:
            raise ValueError("Run analysis before printing summary")

        print("\n=== Overall Performance ===")
        self._print_metrics("Strategy", self.results['overall'])

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
            print(f"\nAttempt {attempt}:")
            self._print_metrics("Performance", metrics)
            
        # Add Asian session analysis
        self.analyze_missing_asian_days()

    def analyze_missing_asian_days(self):
        """Analyze days where Asian session had no trades"""
        all_trading_days = set(self.trades_df['date'].unique())
        asian_trading_days = set(
            self.trades_df[self.trades_df['session'] == 'asian']['date'].unique()
        )
        
        missing_days = all_trading_days - asian_trading_days
        
        print("\n=== Missing Asian Session Days ===")
        print(f"Total Missing Days: {len(missing_days)}")
        
        if missing_days:
            print("\nDetails for days without Asian session trades:")
            for date in sorted(missing_days):
                day_trades = self.trades_df[self.trades_df['date'] == date]
                other_sessions = day_trades['session'].unique()
                print(f"\nDate: {date}")
                print(f"Other sessions active: {', '.join(other_sessions)}")
                
                session_breakdown = day_trades.groupby('session')['pnl'].agg([
                    ('trades', 'size'),
                    ('pnl', 'sum')
                ])
                
                for session, row in session_breakdown.iterrows():
                    print(f"{session}: {row['trades']} trades, PnL=${row['pnl']:.2f}")

    def _print_metrics(self, label: str, metrics: ReturnMetrics):
        """Helper to print formatted metrics"""
        print(f"{label}:")
        print(f"  Total PnL: ${metrics.total_pnl:.2f}")
        print(f"  Win Rate: {metrics.win_rate:.1%}")
        print(f"  Avg Win: ${metrics.avg_win:.2f}")
        print(f"  Avg Loss: ${metrics.avg_loss:.2f}")
        if metrics.sharpe is not None:
            print(f"  Sharpe Ratio: {metrics.sharpe:.2f}")
        print(f"  Profitable/Losing Days: {metrics.profitable_days}/{metrics.losing_days}")

if __name__ == '__main__':
    # Basic setup and execution
    from strategy import TradingStrategy
    from data_handler import DukasCopyDataHandler
    
    logging.basicConfig(level=logging.INFO)
    
    data_path = Path("datasets/raw/XAUUSD/Nov23-Nov24/XAUUSD_1M_BID.csv")
    handler = DukasCopyDataHandler()
    data = handler.load_data(data_path)
    
    strategy = TradingStrategy(data)
    strategy.generate_signals()
    strategy.simulate_trades()
    
    backtest = Backtest(strategy)
    backtest.run_analysis()
    backtest.print_summary()
    backtest.plot_analysis()