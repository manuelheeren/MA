from typing import Protocol
import pandas as pd

class BetSizingStrategy(Protocol):
    def compute_position(self, capital: float, price: float, stop_loss: float) -> tuple:
        ...

class KellyBetSizing:
    def __init__(self, returns: pd.Series):
        self.kelly_fraction = self._compute_kelly_from_returns(returns)

    def _compute_kelly_from_returns(self, returns: pd.Series) -> float:
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        p = len(positive_returns) / len(returns)
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 0.0
        win_avg = positive_returns.mean()
        loss_avg = abs(negative_returns.mean())
        win_loss_ratio = win_avg / loss_avg
        kelly_fraction = p - ((1 - p) / win_loss_ratio)
        return max(0, min(kelly_fraction, 1))

    def compute_position(self, capital: float, price: float, stop_loss: float) -> tuple:
        risk = abs(price - stop_loss)
        risk_amount = capital * self.kelly_fraction
        position_size = risk_amount / risk if risk != 0 else 0
        return position_size, risk_amount

class FixedFractionalBetSizing:
    def __init__(self, risk_fraction: float = 0.01):
        self.risk_fraction = risk_fraction

    def compute_position(self, capital: float, price: float, stop_loss: float) -> tuple:
        risk = abs(price - stop_loss)
        risk_amount = capital * self.risk_fraction
        position_size = risk_amount / risk if risk != 0 else 0
        return position_size, risk_amount

class FixedBetSize:
    def __init__(self, fixed_risk: float = 500):
        self.fixed_risk = fixed_risk  # USD risk per trade

    def compute_position(self, capital: float, price: float, stop_loss: float) -> tuple:
        risk_per_unit = abs(price - stop_loss)
        position_size = self.fixed_risk / risk_per_unit if risk_per_unit != 0 else 0
        return position_size, self.fixed_risk

