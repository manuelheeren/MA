from new_strategy import TradingStrategy, TradeSetup
import pandas as pd
import numpy as np

class MetaLabelingStrategy(TradingStrategy):
    def __init__(self, data, asset, bet_sizing, bet_sizing_method, meta_model_handler=None, feature_cols=None):
        super().__init__(data, asset, bet_sizing, bet_sizing_method)
        self.meta_model_handler = meta_model_handler
        self.features_to_use = feature_cols or []
        self.rejected_trades = 0

    def _create_trade_setup(self, entry_time, price, direction, attempt, ref_close, session):
        entry_time = pd.to_datetime(entry_time).tz_convert("UTC").floor("min")

        stop_loss, take_profit = self._calculate_trade_levels(price, direction, attempt)
        current_equity = self._get_session_equity(session, price)
        available_cash = self._get_session_available_cash(session)

        # Base context
        context = {
            "attempt": attempt,
            "ref_close": ref_close,
            "duration_minutes": 0,
            "session": session
        }

        # Always call compute_position once
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
            # Fallback for older implementations
            result = self.bet_sizing.compute_position(
                equity=current_equity,
                price=price,
                stop_loss=stop_loss
            )

        # Handle both 2-tuple and 3-tuple responses
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

        # Merge enriched context
        context.update(enriched_context)

        # Use meta-model to approve/reject
        if self.meta_model_handler:
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
