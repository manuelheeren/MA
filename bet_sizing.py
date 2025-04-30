from typing import Protocol
import pandas as pd

class BetSizingStrategy(Protocol):
    def compute_position(self, capital: float, price: float, stop_loss: float) -> tuple:
        ...

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

class KellyBetSizing:
    def __init__(self, min_trades: int = 20, risk_pct: float = 0.01):
        self.min_trades = min_trades
        self.risk_pct = risk_pct  # 1% default for early phase
        self.trade_returns = []  # Always a plain Python list
        self.kelly_fraction = 0.0

    def update_with_trade_result(self, pnl: float, risk_amount: float):
        if risk_amount > 0:
            self.trade_returns.append(pnl / risk_amount)
            if len(self.trade_returns) > self.min_trades:
                self.trade_returns.pop(0)

    def _compute_kelly_fraction(self) -> float:
        if len(self.trade_returns) < self.min_trades:
            return 0.0

        returns = pd.Series(self.trade_returns)
        pos = returns[returns > 0]
        neg = returns[returns < 0]

        if len(pos) == 0 or len(neg) == 0:
            return 0.0

        p = len(pos) / len(returns)
        win_avg = pos.mean()
        loss_avg = abs(neg.mean())

        ratio = win_avg / loss_avg if loss_avg != 0 else 0
        kelly = p - ((1 - p) / ratio) if ratio != 0 else 0

        return max(0, min(kelly, 1))

    def compute_position(self, capital: float, price: float, stop_loss: float) -> tuple:
        risk_per_unit = abs(price - stop_loss)

        # Use fixed risk approach (1% of capital with stop-loss-aware sizing)
        if len(self.trade_returns) < self.min_trades:
            position_size = (capital * self.risk_pct) / risk_per_unit if risk_per_unit != 0 else 0
            risk_amount = risk_per_unit * position_size

            # Failsafe: fallback if calculation explodes
            if not (np.isfinite(position_size).all() and np.isfinite(risk_amount).all()):
                position_size = (capital * self.risk_pct) / (price * 0.01)
                risk_amount = capital * self.risk_pct

            return position_size, risk_amount

        # Use Kelly fraction after enough trade history
        self.kelly_fraction = self._compute_kelly_fraction()
        risk_amount = capital * self.kelly_fraction
        position_size = risk_amount / risk_per_unit if risk_per_unit != 0 else 0

        return position_size, risk_amount

class FixedFractionalBetSizing:
    def __init__(self, investment_fraction: float = 0.01):
        self.investment_fraction = investment_fraction  # e.g., 0.01 for 1% of equity

    def compute_position(self, equity: float, price: float, stop_loss: float) -> tuple:
        amount_to_invest = equity * self.investment_fraction
        position_size = amount_to_invest / price if price != 0 else 0
        return position_size, amount_to_invest

class FixedBetSize:
    def __init__(self, fixed_trade_size: float = 2000):
        self.fixed_trade_size = fixed_trade_size

    def compute_position(self, capital: float, price: float, stop_loss: float) -> tuple:
        # Capital check
        if capital < self.fixed_trade_size:
            return 0, 0  # Cannot afford trade

        position_size = self.fixed_trade_size / price if price != 0 else 0
        return position_size, self.fixed_trade_size


