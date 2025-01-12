from pathlib import Path
import pandas as pd
import logging
from typing import Dict, List
import numpy as np

class HoldingTimeAnalyzer:
    """Analyzes trade holding times from backtest results."""
    
    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or Path("datasets/results")
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        return logger
        
    def load_trade_data(self, asset: str) -> pd.DataFrame:
        """Load trade data for specified asset."""
        file_path = self.results_dir / f"trades_detailed_{asset}.csv"
        if not file_path.exists():
            self.logger.error(f"Trade data file not found: {file_path}")
            return pd.DataFrame()
            
        df = pd.read_csv(file_path)
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        return df
        
    def analyze_holding_times(self, df: pd.DataFrame) -> Dict:
        """Analyze holding times per session and attempt."""
        results = {}
        
        # Analyze per session
        for session in ['asian', 'london', 'us']:  # Fixed order
            session_data = df[df['session'] == session]
            results[session] = {
                'overall': self._calculate_duration_stats(session_data),
                'attempts': {}
            }
            
            # Analyze per attempt within session
            for attempt in sorted(session_data['attempt'].unique()):
                attempt_data = session_data[session_data['attempt'] == attempt]
                results[session]['attempts'][attempt] = self._calculate_duration_stats(attempt_data)
                
        return results
    
    def _calculate_duration_stats(self, data: pd.DataFrame) -> Dict:
        """Calculate duration statistics for a dataset."""
        if data.empty:
            return {
                'count': 0,
                'min': None,
                'max': None,
                'mean': None,
                'median': None,
                'std': None
            }
            
        durations = data['duration_minutes']
        return {
            'count': len(durations),
            'min': durations.min(),
            'max': durations.max(),
            'mean': durations.mean(),
            'median': durations.median(),
            'std': durations.std()
        }
    
    def print_analysis(self, results: Dict, asset: str) -> None:
        """Print formatted analysis results with side-by-side comparison."""
        sessions = ['asian', 'london', 'us']
        
        # Print asset name
        print(f"\n{'='*80}")
        print(f"{asset} Analysis")
        print(f"{'='*80}")
        
        # Print overall statistics
        print("\nOverall Statistics:")
        print("-" * 80)
        
        # Header for sessions
        print(f"{'Metric':<15} {'Asian':>20} {'London':>20} {'US':>20}")
        print("-" * 80)
        
        # Print each metric
        metrics = ['count', 'min', 'max', 'mean', 'median', 'std']
        metric_names = {
            'count': 'Trades',
            'min': 'Min (min)',
            'max': 'Max (min)',
            'mean': 'Mean (min)',
            'median': 'Median (min)',
            'std': 'Std Dev (min)'
        }
        
        for metric in metrics:
            values = [results[session]['overall'][metric] for session in sessions]
            if metric == 'count':
                formatted_values = [f"{v:d}" if v is not None else "N/A" for v in values]
            else:
                formatted_values = [f"{v:.2f}" if v is not None else "N/A" for v in values]
            print(f"{metric_names[metric]:<15} " + "".join(f"{val:>20}" for val in formatted_values))
            
        # Find maximum attempt number across all sessions
        attempt_numbers = []
        for session in sessions:
            attempt_numbers.extend(results[session]['attempts'].keys())
        max_attempts = max(attempt_numbers) if attempt_numbers else 0
        
        # Print per-attempt statistics
        for attempt in range(1, max_attempts + 1):
            print(f"\nAttempt {attempt} Statistics:")
            print("-" * 80)
            print(f"{'Metric':<15} {'Asian':>20} {'London':>20} {'US':>20}")
            print("-" * 80)
            
            for metric in metrics:
                values = [
                    results[session]['attempts'].get(attempt, {'count': 0})[metric]
                    if attempt in results[session]['attempts']
                    else None
                    for session in sessions
                ]
                if metric == 'count':
                    formatted_values = [f"{v:d}" if v is not None else "N/A" for v in values]
                else:
                    formatted_values = [f"{v:.2f}" if v is not None else "N/A" for v in values]
                print(f"{metric_names[metric]:<15} " + "".join(f"{val:>20}" for val in formatted_values))

def main():
    """Main function to run holding time analysis."""
    analyzer = HoldingTimeAnalyzer()
    
    # List of assets to analyze
    assets = ["XAUUSD", "BTCUSD"]
    
    for asset in assets:
        # Load and analyze data
        df = analyzer.load_trade_data(asset)
        if df.empty:
            print(f"No data found for {asset}")
            continue
            
        # Perform analysis and print results
        results = analyzer.analyze_holding_times(df)
        analyzer.print_analysis(results, asset)

if __name__ == "__main__":
    main()