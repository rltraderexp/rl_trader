import pandas as pd
import numpy as np
from rl_trading_project.features.basic_features import assemble_features

def make_series(n=100):
    rng = np.random.RandomState(42)
    price = 100 + np.cumsum(rng.normal(scale=0.2, size=n))
    # FIX: 'T' is deprecated, use 'min' instead
    dates = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price,
        'high': price + 0.1,
        'low': price - 0.1,
        'close': price,
        'volume': rng.randint(1,100, size=n)
    })
    return df

if __name__ == '__main__':
    df = make_series(60)
    out_df = assemble_features(df)
    
    print("DataFrame with calculated features (tail):")
    print(out_df[['timestamp','close','ret_1','ma','ma_z','atr','rsi']].tail(10))