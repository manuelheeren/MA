# Time Series Momentum Trading Strategy

Implementation and backtesting of a momentum-based trading strategy analyzing market regimes and trading sessions.

## Overview

The strategy trades based on price momentum relative to the previous day's close, with session-specific analysis and market regime classification.

## Structure

- `backtest.py`: Core backtesting engine
- `strategy.py`: Trading strategy implementation
- `data_handler.py`: Data loading and processing
- `market_analysis.py`: Market regime classification and analysis
- `util_volume_analysis.py`: Volume analysis utilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-strategy.git
cd trading-strategy
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn
```

## Data Format

Expected data structure (CSV):
- Minute-level OHLCV data
- Columns: timestamp, open, high, low, close, volume
- Example path: `datasets/raw/XAUUSD/Nov23-Nov24/XAUUSD_1M_BID.csv`

## Usage

Run backtest:
```bash
python backtest.py
```

Analyze market regimes:
```bash
python market_analysis.py
```

Analyze volume patterns:
```bash
python util_volume_analysis.py
```

## Output

- Trading performance metrics
- Session analysis
- Market regime classification
- Performance visualizations

## License

TBD