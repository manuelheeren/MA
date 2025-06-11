from new_strategy import TradingStrategy, TradeSetup
import pandas as pd
import numpy as np
from collections import deque
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

        context = {
            "attempt": attempt,
            "ref_close": ref_close,
            "duration_minutes": 0,
            "session": session
        }

        feature_cols = ['atr_14', 'ma_14', 'min_price_30', 'max_price_30']
        if entry_time in self.data.index:
            for col in feature_cols:
                context[col] = self.data.at[entry_time, col]
        else:
            for col in feature_cols:
                context[col] = None

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

        context.update(enriched_context)

        #if self.rolling_evaluator:
          #  live_metrics = self.rolling_evaluator.get_metrics()
           # for k, v in live_metrics.items():
              #  context[f"rolling_{k}"] = v

        print(f"[Eval Input] {entry_time} | Direction: {direction} | Context: {context}")

        # Debug feature values
        missing_features = [k for k in self.features_to_use if pd.isna(context.get(k)) or context.get(k) in [np.inf, -np.inf]]
        if missing_features:
            print(f"[SKIP: Invalid Features] {entry_time} | Missing or invalid: {missing_features}")

        if self.meta_model_handler:
            approved = self.meta_model_handler.is_trade_approved(context, direction)
            if not approved:
                print(f"[SKIP] Trade at {entry_time} rejected by meta model")
                self.rejected_trades += 1
                return None

            #predicted_label = 1 if direction == 'long' else 0
            #true_label = (
              #  1 if (direction == 'long' and price < stop_loss) or (direction == 'short' and price > stop_loss)
              #  else 0
           # )
           # self.rolling_evaluator.update(true_label, predicted_label)


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

