# SVI fitting demo: generate synthetic quotes with volatility smile and fit SVI surface
import numpy as np
import pandas as pd
from .svi_surface import SviSurface
from .iv_surface import IVSurface
from .black_scholes import bs_price

def make_series(n=400, seed=2):
    rng = np.random.RandomState(seed)
    price = 100 + np.cumsum(rng.normal(scale=0.2, size=n))
    dates = pd.date_range('2025-01-01', periods=n, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp':dates, 'open':price, 'high':price+0.1, 'low':price-0.1, 'close':price, 'volume':rng.randint(1,100,size=n)})
    return df

def synth_quotes(S, strikes, taus):
    quotes = []
    for tau in taus:
        for K in strikes:
            # synthetic vol: base 0.18, add smile component
            m = np.log(K/S)
            iv = 0.18 + 0.05 * np.exp(- (m**2)/0.02) + 0.01 * (1.0/ (1.0 + 5*tau))
            price = bs_price(S, K, r=0.0, sigma=iv, tau=tau, option_type='call')
            quotes.append({'strike': K, 'tau': tau, 'price': price, 'type': 'call', 'spot': S, 'r': 0.0})
    return quotes

if __name__ == '__main__':
    try:
        from scipy.optimize import least_squares
    except ImportError:
        print("Scipy not found, skipping SVI demo.")
    else:
        df = make_series(400)
        S = float(df['close'].iloc[-1])
        strikes = np.linspace(80, 120, 9)
        taus = np.array([30, 60, 90]) / 252.0
        
        quotes = synth_quotes(S, strikes, taus)
        
        svi = SviSurface.from_quotes(quotes)
        
        print("Fitted SVI Parameters per Tau:")
        for t in svi.taus:
            params = svi.params_by_tau[t]
            print(f"tau={t:.4f}: a={params[0]:.4f}, b={params[1]:.4f}, rho={params[2]:.4f}, m={params[3]:.4f}, sigma={params[4]:.4f}")
            
        print("\nIV at K=100, tau=90/252:", svi.interp_iv(100.0, 90/252))
        print("IV at K=110, tau=60/252:", svi.interp_iv(110.0, 60/252))