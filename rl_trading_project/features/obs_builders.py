"""Observation builders that convert env.window DataFrame into feature arrays for agents."""
import numpy as np
import pandas as pd
from .basic_features import assemble_features

def window_features_df(df_window: pd.DataFrame) -> np.ndarray:
    """
    Given a small canonical OHLCV df window, compute and return a 1D numpy array of features.
    Uses assemble_features and returns last-row indicators: rsi, atr, ma_z, ret_1
    """
    if df_window.empty:
        return np.zeros(5, dtype=np.float32) # Return a zero vector if the window is empty
        
    df = assemble_features(df_window)
    last = df.iloc[-1]
    
    feats = np.array([
        float(last.get('ret_1', 0.0)),
        float(last.get('ma', 0.0)),
        float(last.get('ma_z', 0.0)),
        float(last.get('atr', 0.0)),
        float(last.get('rsi', 0.0))
    ], dtype=np.float32)
    
    return np.nan_to_num(feats) # Ensure no NaNs are returned