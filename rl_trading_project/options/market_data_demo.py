# Market data ingestion demo - synthetic option chain and building quotes for IV surface
import numpy as np
import pandas as pd
import datetime as dt
from .market_data import normalize_chain, chain_to_quotes, estimate_forward_from_chain
from .black_scholes import bs_price

def make_chain(S=100.0, strikes=None, taus=None):
    if strikes is None:
        strikes = [90, 95, 100, 105, 110]
    if taus is None:
        taus = [30/365.25, 60/365.25, 90/365.25]  # in years
    
    rows = []
    timestamp = pd.Timestamp('2025-08-01T12:00:00Z')
    for tau in taus:
        expiry = timestamp + pd.Timedelta(days=int(tau * 365.25))
        for K in strikes:
            # synthetic vol with smile
            m = np.log(K / S)
            vol = 0.20 + 0.10 * (m**2) + 0.05 * np.exp(-tau * 5)
            
            price_call = bs_price(S, K, r=0.01, sigma=vol, tau=tau, option_type='call')
            price_put = bs_price(S, K, r=0.01, sigma=vol, tau=tau, option_type='put')
            
            # create bid/ask around mid with a spread
            spread = 0.05 * price_call
            rows.append({
                'strike': K, 
                'expiry': expiry, 
                'timestamp': timestamp, 
                'call_bid': max(0.01, price_call - spread / 2), 
                'call_ask': price_call + spread / 2, 
                'put_bid': max(0.01, price_put - spread / 2), 
                'put_ask': price_put + spread / 2, 
                'underlying_price': S,
                'r': 0.01
            })
    return pd.DataFrame(rows)

if __name__ == '__main__':
    df = make_chain(S=102.5)
    print("Original Chain Preview:\n", df.head())
    
    norm = normalize_chain(df)
    print("\nNormalized Chain Preview:\n", norm.head())
    
    F, q = estimate_forward_from_chain(norm, assume_r=0.01)
    print(f"\nEstimated Forward Price: {F:.4f}, Implied Dividend Yield: {q:.4f}")
    
    quotes = chain_to_quotes(norm, assume_r=0.01)
    print("\nNumber of quotes generated:", len(quotes))
    if quotes:
        print("Sample quote:", quotes[0])