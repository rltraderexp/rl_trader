"""Market data ingestion utilities for option chains.

Provides functions to:
- normalize option chain DataFrame (compute mid prices, filter)
- estimate implied forward price and dividend yield
- convert option chain rows into quote dicts suitable for IVSurface/SviSurface

Expected input DataFrame columns (flexible):
- strike, call_bid, call_ask, put_bid, put_ask, underlying_price, expiry (datetime), timestamp (datetime), r (optional risk-free rate)
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import math
from pandas.api.types import is_datetime64_any_dtype

def normalize_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mid prices for calls and puts, drop rows without both sides or with nonpositive mid price."""
    d = df.copy()
    
    def get_mid_price(bid_col, ask_col, price_col):
        if bid_col in d.columns and ask_col in d.columns:
            return (d[bid_col].astype(float) + d[ask_col].astype(float)) / 2.0
        elif price_col in d.columns:
            return d[price_col].astype(float)
        return pd.Series(np.nan, index=d.index)

    d['call_mid'] = get_mid_price('call_bid', 'call_ask', 'call_price')
    d['put_mid'] = get_mid_price('put_bid', 'put_ask', 'put_price')

    # Standardize other column names
    if 'underlying_price' not in d.columns and 'spot' in d.columns:
        d['underlying_price'] = d['spot']
    if 'strike' not in d.columns:
        raise ValueError("Input chain must contain 'strike' column.")

    # Ensure timestamps are in datetime format
    for col in ['expiry', 'timestamp']:
        if col in d.columns and not is_datetime64_any_dtype(d[col]):
            d[col] = pd.to_datetime(d[col], errors='coerce')

    # Filter out invalid rows
    d.dropna(subset=['call_mid', 'put_mid'], how='all', inplace=True)
    d = d[(d['call_mid'].isna() | (d['call_mid'] > 0)) & (d['put_mid'].isna() | (d['put_mid'] > 0))]
    return d.reset_index(drop=True)

def estimate_forward_from_chain(df: pd.DataFrame, assume_r: Optional[float]=None) -> Tuple[float, float]:
    """
    Estimate forward price F and implied dividend yield q using put-call parity across strikes.
    Returns (F_median, q_median). If r is provided, uses it in parity; otherwise assumes r=0.
    Parity: C - P = S*e^{-qT} - K*e^{-rT}
    Implied Forward: F = K + e^{rT} * (C - P)
    """
    d = df.copy()
    if 'underlying_price' not in d.columns:
        raise ValueError("Chain must include 'underlying_price' or 'spot' column.")
        
    S = float(d['underlying_price'].dropna().iloc[0])
    
    if 'expiry' in d.columns and 'timestamp' in d.columns:
        Ts = ((d['expiry'] - d['timestamp']).dt.total_seconds() / (365.25 * 24 * 3600)).astype(float)
    elif 'time_to_expiry' in d.columns:
        Ts = d['time_to_expiry'].astype(float)
    else:
        Ts = pd.Series([7.0/252.0]*len(d), index=d.index)
    d['T'] = Ts.clip(lower=1e-8)
    
    r = float(assume_r) if assume_r is not None else 0.0

    # Calculate implied forward for each strike using put-call parity
    parity_df = d.dropna(subset=['call_mid', 'put_mid']).copy()
    parity_df['F'] = parity_df['strike'] + np.exp(r * parity_df['T']) * (parity_df['call_mid'] - parity_df['put_mid'])
    
    # Calculate implied dividend yield from the forward price
    # F = S * e^{(r-q)T} => q = r - (1/T) * ln(F/S)
    parity_df['q'] = r - (1 / parity_df['T']) * np.log(parity_df['F'] / S)
    
    if parity_df.empty:
        return S, 0.0 # Fallback

    # Return robust median estimates
    F_med = float(np.nanmedian(parity_df['F']))
    q_med = float(np.nanmedian(parity_df['q']))
    return F_med, q_med

def chain_to_quotes(df: pd.DataFrame, assume_r: Optional[float]=None) -> List[Dict[str,Any]]:
    """
    Convert normalized option chain DataFrame to a list of quote dicts suitable for IVSurface/SviSurface.
    """
    d = normalize_chain(df)
    
    S = float(d['underlying_price'].dropna().iloc[0])
    r = float(assume_r) if assume_r is not None else 0.0

    if 'expiry' in d.columns and 'timestamp' in d.columns:
        d['tau'] = ((d['expiry'] - d['timestamp']).dt.total_seconds() / (365.25 * 24 * 3600)).astype(float)
    elif 'time_to_expiry' in d.columns:
        d['tau'] = d['time_to_expiry'].astype(float)
    else:
        d['tau'] = 7.0/252.0
    d['tau'] = d['tau'].clip(lower=1e-8)
        
    quotes = []
    for _, row in d.iterrows():
        quote_base = {
            'strike': float(row['strike']),
            'tau': float(row['tau']),
            'spot': S,
            'r': r
        }
        if pd.notna(row['call_mid']):
            quotes.append({**quote_base, 'price': float(row['call_mid']), 'type': 'call'})
        if pd.notna(row['put_mid']):
            quotes.append({**quote_base, 'price': float(row['put_mid']), 'type': 'put'})
            
    return quotes