"""
Basic feature engineering utilities for OHLCV time-series.

Provides:
- pct_change returns
- rolling mean/std
- ATR (Average True Range)
- RSI (Relative Strength Index)
- z-score normalization over rolling window
- assemble_features(df, window) -> df with added columns
"""

from typing import Optional
import pandas as pd
import numpy as np

def pct_change(df: pd.DataFrame, col: str = 'close', periods: int = 1) -> pd.Series:
    return df[col].pct_change(periods=periods)

def rolling_mean(df: pd.DataFrame, col: str = 'close', window: int = 14) -> pd.Series:
    return df[col].rolling(window=window, min_periods=1).mean()

def rolling_std(df: pd.DataFrame, col: str = 'close', window: int = 14) -> pd.Series:
    return df[col].rolling(window=window, min_periods=1).std()

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False).mean() # Use exponential moving average for smoother ATR

def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # Use exponential moving average for RSI calculation
    ma_up = up.ewm(com=window - 1, adjust=True, min_periods=window).mean()
    ma_down = down.ewm(com=window - 1, adjust=True, min_periods=window).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def zscore(df: pd.DataFrame, col: str = 'close', window: int = 20) -> pd.Series:
    roll_mean = df[col].rolling(window=window, min_periods=1).mean()
    roll_std = df[col].rolling(window=window, min_periods=1).std().replace(0, 1e-9)
    return (df[col] - roll_mean) / roll_std

def assemble_features(df: pd.DataFrame, windows: Optional[dict] = None) -> pd.DataFrame:
    """
    Adds multiple derived features to a copy of df and returns it.
    windows: dict specifying window sizes for indicators, e.g. {'rsi':14, 'z':20, 'atr':14}
    """
    if windows is None:
        windows = {'rsi':14, 'z':20, 'atr':14, 'ret':1, 'ma':10}
    out = df.copy()
    out['ret_1'] = pct_change(out, 'close', periods=windows.get('ret',1))
    out['ma'] = rolling_mean(out, 'close', window=windows.get('ma',10))
    out['ma_z'] = zscore(out, 'close', window=windows.get('z',20))
    out['atr'] = atr(out, window=windows.get('atr',14))
    out['rsi'] = rsi(out, window=windows.get('rsi',14))
    # fill NaNs resulting from rolling windows
    # FIX: Use obj.bfill() and obj.ffill() instead of the 'method' argument
    out = out.bfill().ffill().fillna(0)
    return out