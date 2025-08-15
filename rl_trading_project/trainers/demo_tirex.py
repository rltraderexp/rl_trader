# Demo for TiRexAdapter usage (falls back to stub if TiRex is not installed)
import pandas as pd
import numpy as np
from ..forecasting.tirex_adapter import TiRexAdapter

def make_series(n=200):
    rng = np.random.RandomState(1)
    price = 100 + np.cumsum(rng.normal(scale=0.2, size=n))
    dates = pd.date_range('2025-01-01', periods=n, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp':dates, 'open':price+0.1, 'high':price+0.2, 'low':price-0.2, 'close':price, 'volume':rng.randint(1,100,size=n)})
    return df

if __name__ == '__main__':
    df = make_series(300)
    
    # Initialize adapter. It will use the stub since model_path is None.
    adapter = TiRexAdapter(model_path=None) 
    
    # Get a forecast
    history_df = df.iloc[:120]
    preds = adapter.predict(history_df, horizon=30)
    
    print("Forecast generated using TiRexAdapter (stub).")
    print("Mean forecast (first 5 steps):", preds['mean'][:5])
    print("Median forecast (first 5 steps):", preds['quantiles']['0.5'][:5])
    print("10% quantile (first 5 steps):", preds['quantiles']['0.1'][:5])
    print("90% quantile (first 5 steps):", preds['quantiles']['0.9'][:5])