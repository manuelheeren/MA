# Global Feature Sets from SHAP Analysis

# === UNIFIED FEATURE SETS ===
top_15_features = ['attempt', 'ret_30m', 'ret30m_voladj', 'pos_in_day_range', 'range15m_voladj', 'avg_return_30d', 'volume_shifted', 'rolling_f1', 'ma14_slope_5', 'atr_z_60', 'daily_volatility', 'dtwexbgs', 'open', 'drawdown_30', 'vol_15m']

top_10_features = ['attempt', 'ret_30m', 'ret30m_voladj', 'pos_in_day_range', 'range15m_voladj', 'avg_return_30d', 'volume_shifted', 'rolling_f1', 'ma14_slope_5', 'atr_z_60']

top_8_features = ['attempt', 'ret_30m', 'ret30m_voladj', 'pos_in_day_range', 'range15m_voladj', 'avg_return_30d', 'volume_shifted', 'rolling_f1']

conservative_features = ['attempt', 'ret_30m', 'ret30m_voladj', 'pos_in_day_range', 'range15m_voladj', 'avg_return_30d', 'volume_shifted', 'rolling_f1']

# === DIRECTION-SPECIFIC FEATURE SETS ===
long_top_10_features = ['attempt', 'pos_in_day_range', 'daily_volatility', 'rolling_f1', 'volume_shifted', 'atr_z_60', 'dgs10', 'range15m_voladj', 'ret_30m', 'vol_15m']

long_top_8_features = ['attempt', 'pos_in_day_range', 'daily_volatility', 'rolling_f1', 'volume_shifted', 'atr_z_60', 'dgs10', 'range15m_voladj']

short_top_10_features = ['attempt', 'ret30m_voladj', 'avg_return_30d', 'ret_30m', 'range15m_voladj', 'ma14_slope_5', 'volume_shifted', 'atr_z_60', 'cpiaucsl', 'dtwexbgs']

short_top_8_features = ['attempt', 'ret30m_voladj', 'avg_return_30d', 'ret_30m', 'range15m_voladj', 'ma14_slope_5', 'volume_shifted', 'atr_z_60']

