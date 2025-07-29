from new_strategy import TradingStrategy, TradeSetup
import pandas as pd
import numpy as np

class MetaLabelingStrategy(TradingStrategy):
    def __init__(self, data, asset, bet_sizing, bet_sizing_method, meta_model_handler=None, feature_cols=None, rolling_window=30):
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
        "rolling_f1": metrics.get("rolling_f1"),
        "rolling_accuracy": metrics.get("rolling_accuracy"),
        "rolling_precision": metrics.get("rolling_precision"),
        "rolling_recall": metrics.get("rolling_recall"),
        "n_total_seen": metrics.get("n_total_seen"),
        "n_window_obs": metrics.get("n_window_obs"),
        "session_code": session_code,
        }

        # FIX: Add feature columns into context just like base strategy
        feature_cols = ['atr_14', 'ma_14', 'min_price_30', 'max_price_30','daily_return', 'daily_volatility', 't10yie', 'vix_close',"day_of_week", "hour_of_day","dgs10","avg_return_30d","drawdown_30",'close', 'drawdown_static', 'high', 'daily_low', 'week_number', 'max_price_14', 'true_range', 'daily_high', 'low', 'volume', 'dtwexbgs', 'open', 'min_price_14', 'cpiaucsl', 'daily_close','volume_shifted']
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

            # ✅ Inject metrics into context (with fallback to 0.0)
            for key, value in metrics.items():
                context[key] = value if value is not None else 0.0
            print(f"🧠 Injected rolling metrics at {entry_time} ({session}): {metrics}")

            # ✅ Waive trades until window is full
            if metrics.get("n_window_obs", 0) < self.rolling_metrics[session].window_size:
                print(f"🟡 Waving through trade at {entry_time} — not enough metrics yet")
            else:
                # ✅ Apply meta-model when metrics are ready
                if self.meta_model_handler:
                    print(f"\n🧠 Meta Model Context Check ({entry_time}, {session}):")
                    for k, v in context.items():
                        print(f"  {k}: {v}")
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
