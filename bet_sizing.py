from typing import Protocol, Dict
import pandas as pd
import numpy as np

class BetSizingStrategy(Protocol):
    def compute_position(self, capital: float, price: float, stop_loss: float) -> tuple:
        ...

class KellyBetSizing:
    def __init__(self, min_trades: int = 20, risk_pct: float = 0.01):
        self.min_trades = min_trades
        self.risk_pct = risk_pct  # 1% default for early phase
        self.trade_returns = []
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

    def compute_position(self, capital: float, price: float, stop_loss: float, context: dict = None) -> tuple:
        amount_to_invest = capital * self.investment_fraction
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

class PercentVolatilityBetSizing:
    requires_context = True
    def __init__(self, risk_fraction: float = 0.01, atr_column: str = 'atr_14'):
        self.risk_fraction = risk_fraction
        self.atr_column = atr_column

    def compute_position(self, equity: float, price: float, stop_loss: float, context: Dict = None) -> tuple:
        if context is None or self.atr_column not in context:
            return 0, 0  # ATR not available

        atr = context[self.atr_column]
        if atr is None or not np.isfinite(atr) or atr == 0:
            return 0, 0

        risk_amount = equity * self.risk_fraction
        position_size = risk_amount / atr
        return position_size, risk_amount

class OptimalF:
    def __init__(self, min_trades: int = 20, default_fraction: float = 0.01):
        self.min_trades = min_trades
        self.default_fraction = default_fraction
        self.trade_returns = []
        self.optimal_f = default_fraction
        self.total_trades_seen = 0  # ✅ NEW: Tracks total trades processed (not just rolling window)

    def update_with_trade_result(self, pnl: float, risk_amount: float = None):
        self.total_trades_seen += 1  # ✅ Track total trades (for fallback logic)
        self.trade_returns.append(pnl)
        if len(self.trade_returns) > self.min_trades:
            self.trade_returns.pop(0)

    def _compute_optimal_f(self) -> float:
        if len(self.trade_returns) < self.min_trades:
            return self.default_fraction

        biggest_loss = min(self.trade_returns)
        if biggest_loss >= 0:
            return 0.0  # No losses yet, can't compute

        f_values = np.linspace(0.01, 1.0, 100)
        best_f = 0.0
        best_twr = -np.inf

        for f in f_values:
            twr = 1.0
            for trade in self.trade_returns:
                hpr = 1 + f * (trade / abs(biggest_loss))
                if hpr <= 0:
                    twr = -np.inf  # Skip invalid cases
                    break
                twr *= hpr
            if twr > best_twr:
                best_twr = twr
                best_f = f

        return best_f

    def compute_position(self, capital: float, price: float, stop_loss: float) -> tuple:
        # Use total_trades_seen to decide fallback
        if self.total_trades_seen < self.min_trades:
            risk_amount = 1000.0  # Fixed amount for first 20 trades
            position_size = risk_amount / price
            print(f"[OptimalF] (Fallback) capital={capital}, price={price}, RISK_AMOUNT=1000 (fixed fallback)")
            return position_size, risk_amount
        else:
        # Optimal f logic starts after enough trades
            self.optimal_f = self._compute_optimal_f()
            risk_amount = capital * self.optimal_f
            position_size = risk_amount / price
            print(f"[OptimalF] (Optimal f) capital={capital}, optimal_f={self.optimal_f:.4f}, risk_amount={risk_amount:.2f}")
            return position_size, risk_amount

