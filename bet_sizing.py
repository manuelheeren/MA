from typing import Protocol, Dict, Optional
import pandas as pd
import numpy as np

class BetSizingStrategy(Protocol):
    def compute_position(self, capital: float, price: float, stop_loss: float) -> tuple:
        ...

class KellyBetSizing:
    def __init__(self, window_size: int = 100, fallback_fraction: float = 0.02):
        """
        Kelly Criterion with cumulative stats (no rolling window after warmup).

        :param window_size: Number of initial trades to warm up before using Kelly (e.g., 100).
        :param fallback_fraction: Fraction of capital to bet for the first 100 trades (e.g., 0.02 = 2%).
        """
        self.window_size = window_size
        self.fallback_fraction = fallback_fraction
        self.trade_history = []  # Store ALL trades cumulatively
        self.total_trades_seen = 0
        self.limit_hits = 0  # Track how often f > 1 was capped

    def update_with_trade_result(self, pnl: float, risk_amount: float = None):
        """Track the result of each trade (P&L only)."""
        self.total_trades_seen += 1
        self.trade_history.append(pnl)

    def compute_position(self, capital: float, price: float, stop_loss: float) -> tuple:
        """
        Bets using fallback for first N trades, then switches to cumulative Kelly Criterion.
        """
        # === Fallback: Use fixed % risk until we have enough data ===
        if self.total_trades_seen < self.window_size:
            risk_amount = capital * self.fallback_fraction
            print(f"[Kelly] (Fallback) capital={capital:.2f}, price={price:.2f}, risk_amount={risk_amount:.2f}")
            position_size = risk_amount / price
            return position_size, risk_amount

        # === Kelly logic starts here (cumulative stats) ===
        wins = [p for p in self.trade_history if p > 0]
        losses = [p for p in self.trade_history if p < 0]

        p = len(wins) / len(self.trade_history) if self.trade_history else 0
        b1 = np.mean(wins) if wins else 0
        b2 = abs(np.mean(losses)) if losses else 0  # Keep positive

        # Kelly formula (Chen et al. style)
        if b1 == 0 or b2 == 0:
            f = 0.0  # Avoid div by zero
        else:
            f = abs((p * b1 - (1 - p) * b2) / (b1 * b2))

        # Cap f between 0 and 1
        if f > 1.0:
            self.limit_hits += 1
            f_capped = 1.0
        elif f < 0:
            f_capped = 0.0
        else:
            f_capped = f

        risk_amount = capital * f_capped
        position_size = risk_amount / price

        print(f"[Kelly] capital={capital:.2f}, p={p:.2f}, b1={b1:.2f}, b2={b2:.2f}, "
              f"f={f:.4f} (capped={f_capped:.4f}), risk_amount={risk_amount:.2f}")

        return position_size, risk_amount

    def report_limit_hits(self):
        """Report how many times f > 1 was capped during the run."""
        print(f"[Kelly] The cap of f > 1 was hit {self.limit_hits} times during the backtest.")


class FixedFractionalBetSizing:
    def __init__(self, investment_fraction: float = 0.2):
        self.investment_fraction = investment_fraction  # e.g., 0.01 for 1% of equity

    def compute_position(
        self,
        equity: float,
        price: float,
        stop_loss: float,
        context: Optional[Dict] = None,
        available_cash: Optional[float] = None
    ) -> tuple:
        """
        Calculate position size using a fixed fraction of equity.

        Args:
            equity (float): Total equity (used to calculate investment amount).
            price (float): Current asset price.
            stop_loss (float): Stop loss price (not used here).
            context (Optional[Dict]): Extra data (ignored).
            available_cash (Optional[float]): Available cash to cap position size (optional).

        Returns:
            tuple: (position_size, amount_to_invest)
        """
        amount_to_invest = equity * self.investment_fraction
        position_size = amount_to_invest / price if price != 0 else 0

        #  Optional: cap to available_cash if provided (for consistency)
        if available_cash is not None:
            max_position_size = available_cash / price if price != 0 else 0
            if position_size > max_position_size:
                position_size = max_position_size

        return position_size, amount_to_invest


class FixedBetSize:
    def __init__(self, fixed_trade_size: float = 20000):
        self.fixed_trade_size = fixed_trade_size

    def compute_position(
        self,
        equity: float,
        price: float,
        stop_loss: float,
        context: Optional[Dict] = None,
        available_cash: Optional[float] = None
    ) -> tuple:
        """
        Calculate position size using a fixed dollar amount per trade.

        Args:
            equity (float): Total equity (used for capital check).
            price (float): Current asset price.
            stop_loss (float): Stop loss price (not used here).
            context (Optional[Dict]): Extra data (ignored).
            available_cash (Optional[float]): Available cash to cap position size (optional).

        Returns:
            tuple: (position_size, fixed_trade_size)
        """
        # Basic capital check: if not enough equity, skip the trade
        if equity < self.fixed_trade_size:
            return 0, 0

        position_size = self.fixed_trade_size / price if price != 0 else 0

        #  Optional: cap position size to available_cash if provided
        if available_cash is not None:
            max_position_size = available_cash / price if price != 0 else 0
            if position_size > max_position_size:
                position_size = max_position_size

        return position_size, self.fixed_trade_size


class PercentVolatilityBetSizing:
    requires_context = True

    def __init__(self, risk_fraction: float = 0.01, atr_column: str = 'atr_14'):
        self.risk_fraction = risk_fraction
        self.atr_column = atr_column
        self.limit_hit_counter = 0  #  Counter for how often the size limit was hit

    def compute_position(
        self,
        equity: float,
        price: float,
        stop_loss: float,
        context: Dict = None,
        available_cash: Optional[float] = None
    ) -> tuple:
        """
        Calculate position size using the Percent Volatility model.

        Args:
            equity (float): Total equity (used to calculate risk_amount).
            price (float): Current asset price.
            stop_loss (float): Stop loss price (not used here but kept for API compatibility).
            context (Dict): Dictionary containing ATR and other indicators.
            available_cash (Optional[float]): Available cash to cap position size (prevents over-allocation).

        Returns:
            tuple: (position_size, risk_amount, atr)
        """
        if context is None or self.atr_column not in context:
            return 0, 0, None  # ATR not available

        atr = context[self.atr_column]
        if atr is None or not np.isfinite(atr) or atr == 0:
            return 0, 0, atr  # Return ATR (even if invalid) for logging/debugging

        # Calculate risk amount based on total equity
        risk_amount = equity * self.risk_fraction

        # Initial position size based on ATR
        position_size = risk_amount / atr

        # Cap position size to avoid over-leveraging if available_cash is provided
        if available_cash is not None:
            max_position_size = available_cash / price  # max units you can buy/sell with available cash
            if position_size > max_position_size:
                position_size = max_position_size
                self.limit_hit_counter += 1  # Increment counter when cap is applied

        return position_size, risk_amount, atr




class OptimalF:
    def __init__(self, min_trades: int = 20, default_fraction: float = 0.01):
        self.min_trades = min_trades
        self.default_fraction = default_fraction
        self.trade_returns = []
        self.optimal_f = default_fraction
        self.total_trades_seen = 0
        self.limit_hit_counter = 0  # ✅ Tracks how often position size was capped

    def update_with_trade_result(self, pnl: float, risk_amount: float = None):
        self.total_trades_seen += 1
        self.trade_returns.append(pnl)
        if len(self.trade_returns) > self.min_trades:
            self.trade_returns.pop(0)

    def _compute_optimal_f(self) -> float:
        if len(self.trade_returns) < self.min_trades:
            return self.default_fraction

        biggest_loss = min(self.trade_returns)
        if biggest_loss >= 0:
            return 0.0  # No losses yet

        f_values = np.linspace(0.01, 1.0, 100)
        best_f = 0.0
        best_twr = -np.inf

        for f in f_values:
            twr = 1.0
            for trade in self.trade_returns:
                hpr = 1 + f * (trade / abs(biggest_loss))
                if hpr <= 0:
                    twr = -np.inf
                    break
                twr *= hpr
            if twr > best_twr:
                best_twr = twr
                best_f = f

        return best_f

    def compute_position(
        self,
        equity: float,
        price: float,
        stop_loss: float,
        context: Optional[Dict] = None,
        available_cash: Optional[float] = None
    ) -> tuple:
        if self.total_trades_seen < self.min_trades:
            risk_amount = 1000.0
            position_size = risk_amount / price if price != 0 else 0
            print(f"[OptimalF] (Fallback) equity={equity}, price={price}, RISK_AMOUNT=1000 (fixed fallback)")
        else:
            self.optimal_f = self._compute_optimal_f()
            risk_amount = equity * self.optimal_f
            position_size = risk_amount / price if price != 0 else 0
            print(f"[OptimalF] (Optimal f) equity={equity}, optimal_f={self.optimal_f:.4f}, risk_amount={risk_amount:.2f}")

        # Cap position size if available_cash is provided
        if available_cash is not None:
            max_position_size = available_cash / price if price != 0 else 0
            if position_size > max_position_size:
                position_size = max_position_size
                self.limit_hit_counter += 1  # Increment counter when cap is hit

        return position_size, risk_amount

