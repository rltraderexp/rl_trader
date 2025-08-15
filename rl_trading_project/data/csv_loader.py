"""CSV Loader for OHLCV data.
Exposes `load_csv(path, timestamp_col='timestamp', tz='UTC') -> pd.DataFrame`.
"""
from typing import Optional
import pandas as pd
from pathlib import Path

CANONICAL_COLS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

def _find_columns(df: pd.DataFrame):
    lc = {c.lower(): c for c in df.columns}
    found = {}
    for col in CANONICAL_COLS:
        if col in lc:
            found[col] = lc[col]
        else:
            # try some common aliases
            aliases = {
                'timestamp': ['time','date','datetime','t'],
                'volume': ['vol']
            }
            for a in aliases.get(col, []):
                if a in lc:
                    found[col] = lc[a]
                    break
    return found

def load_csv(path: str, timestamp_col: str = 'timestamp', tz: Optional[str]='UTC') -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    df = pd.read_csv(path)
    cols = _find_columns(df)
    
    # If timestamp column is not found by alias, use the first column as a default
    if 'timestamp' not in cols and df.columns.any():
        cols['timestamp'] = df.columns[0]
        
    df = df.rename(columns={cols[k]: k for k in cols})
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        if tz is not None:
            try:
                df['timestamp'] = df['timestamp'].dt.tz_convert(tz)
            except TypeError: # This happens if timezone is already set
                df['timestamp'] = df['timestamp'].dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df