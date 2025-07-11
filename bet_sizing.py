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
        self.cash_cap_hits = 0  # Track how often available_cash capped position size

    def update_with_trade_result(self, pnl: float, risk_amount: float = None):
        """Track the result of each trade (P&L only)."""
        self.total_trades_seen += 1
        self.trade_history.append(pnl)

    def compute_position(
        self,
        equity: float,
        price: float,
        stop_loss: float,
        context: Optional[Dict] = None,
        available_cash: Optional[float] = None
    ) -> tuple:
        """
        Bets using fallback for first N trades, then switches to cumulative Kelly Criterion.
        Caps position size if available_cash is provided.
        """
        # === Fallback: Use fixed % risk until we have enough data ===
        if self.total_trades_seen < self.window_size:
            risk_amount = equity * self.fallback_fraction
            position_size = risk_amount / price if price != 0 else 0
            print(f"[Kelly] (Fallback) equity={equity:.2f}, price={price:.2f}, risk_amount={risk_amount:.2f}")
        else:
            # === Kelly logic starts here (cumulative stats) ===
            wins = [p for p in self.trade_history if p > 0]
            losses = [p for p in self.trade_history if p < 0]

            p = len(wins) / len(self.trade_history) if self.trade_history else 0
            b1 = np.mean(wins) if wins else 0
            b2 = abs(np.mean(losses)) if losses else 0  # Keep positive

            if b1 == 0 or b2 == 0:
                f = 0.0  # Avoid div by zero
            else:
                f = (p * b1 - (1 - p) * b2) / (b1 * b2)

            # Cap f between 0 and 1
            if f > 1.0:
                self.limit_hits += 1
                f_capped = 1.0
            elif f < 0:
                f_capped = 0.0
            else:
                f_capped = f

            risk_amount = equity * f_capped
            position_size = risk_amount / price if price != 0 else 0

            print(f"[Kelly] equity={equity:.2f}, p={p:.2f}, b1={b1:.2f}, b2={b2:.2f}, "
                  f"f={f:.4f} (capped={f_capped:.4f}), risk_amount={risk_amount:.2f}")

        # Cap position size if available_cash is provided
        if available_cash is not None:
            max_position_size = available_cash / price if price != 0 else 0
            if position_size > max_position_size:
                position_size = max_position_size
                self.cash_cap_hits += 1  # Increment cash cap counter

        return position_size, risk_amount

    def report_limit_hits(self):
        """Report how many times f > 1 was capped and how often the cash cap was hit."""
        print(f"[Kelly] The cap of f > 1 was hit {self.limit_hits} times during the backtest.")
        print(f"[Kelly] The available cash cap was hit {self.cash_cap_hits} times during the backtest.")



class FixedFractionalBetSizing:
    requires_context = True  

    def __init__(self, investment_fraction: float = 0.2):
        self.investment_fraction = investment_fraction  # e.g., 0.01 for 1% of equity

    def compute_position(
        self,
        equity: float,
        price: float,
        stop_loss: float,
        context: Optional[Dict] = None,
        available_cash: Optional[float] = None,
        session=None
    ) -> tuple:
        """
        Calculate position size using a fixed fraction of equity.
        Returns enriched context for meta-model compatibility.
        """
        amount_to_invest = equity * self.investment_fraction
        position_size = amount_to_invest / price if price != 0 else 0

        # Cap to available cash if provided
        if available_cash is not None:
            max_position_size = available_cash / price if price != 0 else 0
            if position_size > max_position_size:
                position_size = max_position_size

        # ✅ Ensure context has all needed meta features
        if context is None:
            context = {}

        context.update({
            "ma_14": context.get("ma_14"),
            "atr_14": context.get("atr_14"),
            "min_price_30": context.get("min_price_30"),
            "max_price_30": context.get("max_price_30"),
            "session": session,
            "attempt": context.get("attempt"),
            "ref_close": context.get("ref_close")
        })

        # ✅ Return 3-tuple as expected
        return position_size, amount_to_invest, context


class FixedBetSize:
    def __init__(self, fixed_trade_size: float = 20000):
        self.fixed_trade_size = fixed_trade_size

    def compute_position(
        self,
        equity: float,
        price: float,
        stop_loss: float,
        context: Optional[Dict] = None,
        available_cash: Optional[float] = None,
        session = None
    ) -> tuple:
        """
        Calculate position size using a fixed dollar amount per trade.
        Also includes feature data in returned context dict.
        """
        if equity < self.fixed_trade_size:
            return 0, 0, context or {}

        position_size = self.fixed_trade_size / price if price != 0 else 0

        # Cap position size to available cash if needed
        if available_cash is not None:
            max_position_size = available_cash / price if price != 0 else 0
            if position_size > max_position_size:
                position_size = max_position_size

        # 🔁 Return context including all available features (passed in or empty)
        if context is None:
            context = {}

        context.update({
            "ma_14": context.get("ma_14"),
            "atr_14": context.get("atr_14"),
            "min_price_30": context.get("min_price_30"),
            "max_price_30": context.get("max_price_30"),
            "session": session,
            "attempt": context.get("attempt"),
            "ref_close": context.get("ref_close")
        })

        return position_size, self.fixed_trade_size, context



class PercentVolatilityBetSizing:
    requires_context = True

    def __init__(self, risk_fraction: float = 0.01, atr_column: str = 'atr_14'):
        self.risk_fraction = risk_fraction
        self.atr_column = atr_column
        self.limit_hit_counter = 0

    def compute_position(
        self,
        equity: float,
        price: float,
        stop_loss: float,
        context: Dict = None,
        available_cash: Optional[float] = None,
        session=None
    ) -> tuple:
        if context is None:
            context = {}

        atr = context.get(self.atr_column)
        if atr is None or not np.isfinite(atr) or atr == 0:
            context[self.atr_column] = None  # explicitly add to context for output
            return 0, 0, context

        risk_amount = equity * self.risk_fraction
        position_size = risk_amount / atr

        if available_cash is not None:
            max_position_size = available_cash / price
            if position_size > max_position_size:
                position_size = max_position_size
                self.limit_hit_counter += 1

        # Enrich context
        context.update({
            "ma_14": context.get("ma_14"),
            "atr_14": context.get("atr_14"),
            "min_price_30": context.get("min_price_30"),
            "max_price_30": context.get("max_price_30"),
            "session": session,
            "attempt": context.get("attempt"),
            "ref_close": context.get("ref_close")
        })

        return position_size, risk_amount, context

class OptimalF:
    requires_context = True

    def __init__(self, min_trades: int = 20, default_fraction: float = 0.01):
        self.min_trades = min_trades
        self.default_fraction = default_fraction
        self.trade_returns_by_session = {}  # Dict: session → list of h values
        self.optimal_f_by_session = {}      # Dict: session → current optimal f
        self.total_trades_seen = 0
        self.limit_hit_counter = 0  # Tracks how often position size was capped
        
        # Debugging
        self.skipped_trades = 0
        self.valid_trades = 0

    def update_with_trade_result(self, pnl: float, risk_amount: float = None, session: str = None):
        """Update the trade history with new trade results for a specific session"""
        self.total_trades_seen += 1

        if risk_amount is None or risk_amount == 0 or session is None:
            self.skipped_trades += 1
            print(f"[OptimalF] Skipped trade (invalid risk_amount={risk_amount} or session={session})")
            return

        if session not in self.trade_returns_by_session:
            self.trade_returns_by_session[session] = []

        self.trade_returns_by_session[session].append(pnl)
        self.valid_trades += 1

        if len(self.trade_returns_by_session[session]) > self.min_trades:
            self.trade_returns_by_session[session].pop(0)

        print(f"[OptimalF] Trade added: session={session}, pnl={pnl:.2f}, total_trades={len(self.trade_returns_by_session[session])}")

    def _compute_optimal_f(self, session: str) -> float:
        trades = self.trade_returns_by_session.get(session, [])

        if len(trades) < self.min_trades:
            print(f"[OptimalF] Not enough trades to compute f for session '{session}' ({len(trades)}/{self.min_trades})")
            return self.default_fraction

        biggest_loss = min(trades)
        if biggest_loss >= 0:
            print(f"[OptimalF] No losses found in session '{session}' — fallback to f = 0.0")
            return 0.0

        f_values = np.linspace(0.01, 1.0, 100)
        best_f = 0.0
        best_twr = -np.inf

        for f in f_values:
            twr = 1.0
            for trade in trades:
                hpr = 1 + f * (trade / abs(biggest_loss))
                if hpr <= 0:
                    twr = -np.inf
                    break
                twr *= hpr
            if twr > best_twr:
                best_twr = twr
                best_f = f

        print(f"[OptimalF] Best f for session '{session}' = {best_f:.4f}")
        return best_f

    def compute_position(
        self,
        equity: float,
        price: float,
        stop_loss: float,
        context: Optional[Dict] = None,
        available_cash: Optional[float] = None,
        session: Optional[str] = None
    ) -> tuple:
        if context is None:
            context = {}

        if session is None:
            print("[OptimalF] Warning: No session provided, using default fraction")
            risk_amount = equity * self.default_fraction
            position_size = risk_amount / price if price != 0 else 0
        else:
            trade_returns = self.trade_returns_by_session.get(session, [])
            print(f"[OptimalF] Session: {session}, available trades: {len(trade_returns)}/{self.min_trades}")

            if len(trade_returns) < self.min_trades:
                risk_amount = equity * self.default_fraction
                position_size = risk_amount / price if price != 0 else 0
                print(f"[OptimalF] (Default) session={session}, equity={equity:.2f}, price={price:.2f}, risk={self.default_fraction:.4f}, risk_amount={risk_amount:.2f}")
            else:
                optimal_f = self._compute_optimal_f(session)
                self.optimal_f_by_session[session] = optimal_f
                risk_amount = equity * optimal_f
                position_size = risk_amount / price if price != 0 else 0
                print(f"[OptimalF] (Optimal) session={session}, equity={equity:.2f}, optimal_f={optimal_f:.4f}, risk_amount={risk_amount:.2f}")

        if available_cash is not None and available_cash > 0:
            max_position_size = available_cash / price if price != 0 else 0
            if position_size > max_position_size:
                original_position = position_size
                position_size = max_position_size
                self.limit_hit_counter += 1
                print(f"[OptimalF] Position size capped: {original_position:.2f} → {position_size:.2f} (hit #{self.limit_hit_counter})")

        # Enrich context for meta-models or logging
        context.update({
            "ma_14": context.get("ma_14"),
            "atr_14": context.get("atr_14"),
            "min_price_30": context.get("min_price_30"),
            "max_price_30": context.get("max_price_30"),
            "session": session,
            "attempt": context.get("attempt"),
            "ref_close": context.get("ref_close")
        })

        return position_size, risk_amount, context

    def report_limit_hits(self):
        print(f"[OptimalF] Statistics:")
        print(f"  - Total trades seen: {self.total_trades_seen}")
        print(f"  - Valid trades used: {self.valid_trades}")
        print(f"  - Skipped trades: {self.skipped_trades}")
        print(f"  - Available cash cap was hit {self.limit_hit_counter} times")
        for session, trades in self.trade_returns_by_session.items():
            optimal_f = self.optimal_f_by_session.get(session, self.default_fraction)
            print(f"  - Session '{session}': {len(trades)} trades, optimal_f={optimal_f:.4f}")

