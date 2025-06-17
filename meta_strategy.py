from new_strategy import TradingStrategy, TradeSetup
import pandas as pd
import numpy as np

class MetaLabelingStrategy(TradingStrategy):
    def __init__(self, data, asset, bet_sizing, bet_sizing_method, meta_model_handler=None, feature_cols=None, rolling_window= 10):
        super().__init__(data, asset, bet_sizing, bet_sizing_method, rolling_window=rolling_window)
        self.meta_model_handler = meta_model_handler
        self.features_to_use = feature_cols or []
        self.rejected_trades = 0

    def _create_trade_setup(self, entry_time, price, direction, attempt, ref_close, session):
        entry_time = pd.to_datetime(entry_time).tz_convert("UTC").floor("min")

        stop_loss, take_profit = self._calculate_trade_levels(price, direction, attempt)
        current_equity = self._get_session_equity(session, price)
        available_cash = self._get_session_available_cash(session)

        metrics = self.rolling_metrics[session].latest()

        session_map = {'asian': 0, 'london': 1, 'us': 2}
        session_code = session_map.get(session, -1)

        context = {
        "attempt": attempt,
        "ref_close": ref_close,
        "duration_minutes": 0,
        "session": session,
        "eval_f1": metrics.get("rolling_f1"),
        "eval_accuracy": metrics.get("rolling_accuracy"),
        "eval_precision": metrics.get("rolling_precision"),
        "eval_recall": metrics.get("rolling_recall"),
        "n_total_seen": metrics.get("n_total_seen"),
        "n_window_obs": metrics.get("n_window_obs"),
        "session_code": session_code,
        }

        # FIX: Add feature columns into context just like base strategy
        feature_cols = ['atr_14', 'ma_14', 'min_price_30', 'max_price_30']
        if entry_time in self.data.index:
            for col in feature_cols:
                context[col] = self.data.at[entry_time, col]
        else:
            for col in feature_cols:
                context[col] = None

        # Call compute_position
        try:
            result = self.bet_sizing.compute_position(
                equity=current_equity,
                price=price,
                stop_loss=stop_loss,
                available_cash=available_cash,
                context=context,
                session=session
            )
        except TypeError:
            result = self.bet_sizing.compute_position(
                equity=current_equity,
                price=price,
                stop_loss=stop_loss
            )

        # Handle result
        if isinstance(result, tuple):
            if len(result) == 3:
                position_size, risk_amount, enriched_context = result
            elif len(result) == 2:
                position_size, risk_amount = result
                enriched_context = context
            else:
                raise ValueError(f"Unexpected number of return values from compute_position: {len(result)}")
        else:
            raise ValueError("compute_position must return a tuple")

        # Merge enriched context back in
        context.update(enriched_context)

        # Meta-model filter
        if hasattr(self, 'rolling_metrics'):
            metrics = self.rolling_metrics[session].latest()
            if all(v is not None for v in metrics.values()):  # only inject if window is filled
                for key, value in metrics.items():
                    context[key] = value
                print(f"🧠 Rolling metrics injected at {entry_time} ({session}): {metrics}")


        """if self.meta_model_handler:
            approved = self.meta_model_handler.is_trade_approved(context, direction)
            if not approved:
                print(f"[SKIP] Trade at {entry_time} rejected by meta model")
                self.rejected_trades += 1
                return None"""

        # Meta-model filter (only apply if rolling window is mature enough)
        if hasattr(self, 'rolling_metrics'):
            metrics = self.rolling_metrics[session].latest()

            # 🔍 Check if the rolling window is filled
            if metrics.get("n_window_obs", 0) < self.rolling_metrics[session].window_size:
                print(f"🟡 Waving through trade at {entry_time} — rolling window not full yet.")
            elif self.meta_model_handler:
                # ✅ Apply the meta model only if window is filled
                approved = self.meta_model_handler.is_trade_approved(context, direction)
                if not approved:
                    print(f"[SKIP] Trade at {entry_time} rejected by meta model")
                    self.rejected_trades += 1
                    return None


        # Build TradeSetup
        return TradeSetup(
            direction=direction,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            attempt=attempt,
            ref_close=ref_close,
            position_size=position_size,
            risk_amount=risk_amount,
            session=session,
            atr_14=context.get("atr_14"),
            ma_14=context.get("ma_14"),
            min_price_30=context.get("min_price_30"),
            max_price_30=context.get("max_price_30")
        )
