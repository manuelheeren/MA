# Time Series Momentum Trading Strategy

Implementation and backtesting of a momentum-based trading strategy analyzing market regimes and trading sessions.

## Overview

The strategy trades based on price momentum relative to the previous day's close, with session-specific analysis and market regime classification.

## Structure

- `data_handler.py`: Data loading and processing
- `strategy.py`: Trading strategy implementation
- `backtest.py`: Core backtesting engine
- `market_analysis.py`: Market regime classification and analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JuliusScheuerer/ASIM_TSMOM_Trading_Strategy.git
cd ASIM_TSMOM_Trading_Strategy
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install pandas numpy
```

## Data Format

Expected data structure (CSV):
- Minute-level OHLCV data
- Columns: timestamp, open, high, low, close, volume
- Example path: `data/raw/XAUUSD/Nov23-Nov24/XAUUSD_1M_BID.csv`

## Usage

Run backtest:
```bash
# In backtest.py line 302 and 303 the test parameters (asset to test and fees) can be defined
python backtest.py
```

Analyze market regimes:
```bash
python market_analysis.py
```

## Output

- Trading performance metrics
- Session analysis
- Market regime classification
