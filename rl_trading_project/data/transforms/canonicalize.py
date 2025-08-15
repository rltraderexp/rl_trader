"""Canonicalize raw OHLCV frames to strict schema and augment features.
"""
from typing import Optional
import pandas as pd
import numpy as np

REQUIRED = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

def canonicalize_ohlcv(df: pd.DataFrame, require_columns: bool=True, tz: Optional[str]='UTC') -> pd.DataFrame:
    df = df.copy()
    
    # Ensure required columns exist, if required
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing and require_columns:
        raise ValueError(f"Missing required columns: {missing}")

    # Coerce numeric types
    for col in ['open','high','low','close','volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Process timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        if tz is not None:
            try:
                df['timestamp'] = df['timestamp'].dt.tz_convert(tz)
            except Exception:
                df['timestamp'] = df['timestamp'].dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
    
    df = df.dropna(subset=['timestamp']).reset_index(drop=True)
    
    # Augment with VWAP and trade_count if they don't exist
    if 'vwap' not in df.columns and all(c in df.columns for c in ['high', 'low', 'close']):
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        df['vwap'] = tp  # simple proxy for VWAP when volume-weighted not available
    if 'trade_count' not in df.columns:
        df['trade_count'] = 1
        
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df