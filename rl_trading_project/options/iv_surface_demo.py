# IV Surface demo: create synthetic quotes, build surface, and query IV and greeks in OptionsEnv
import numpy as np
import pandas as pd
from .iv_surface import IVSurface
from .options_env import OptionsEnv
from .black_scholes import bs_price

def make_series(n=300, seed=1):
    rng = np.random.RandomState(seed)
    price = 100 + np.cumsum(rng.normal(scale=0.3, size=n))
    dates = pd.date_range('2025-01-01', periods=n, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp':dates, 'open':price, 'high':price+0.1, 'low':price-0.1, 'close':price, 'volume':rng.randint(1,100,size=n)})
    return df

def build_synthetic_quotes(S0, strikes, taus):
    # create synthetic market option prices using BS with a slight vol skew
    quotes = []
    for K in strikes:
        for tau in taus:
            # vol increases with moneyness and decreases with tau
            vol = 0.18 + 0.02 * abs(np.log(K / S0)) + 0.01 / (1.0 + tau * 12)
            price = bs_price(S0, K, r=0.0, sigma=vol, tau=tau, option_type='call')
            quotes.append({'strike': K, 'tau': tau, 'price': price, 'type': 'call', 'spot': S0, 'r': 0.0})
    return quotes

if __name__ == '__main__':
    df = make_series(400)
    S0 = df['close'].iloc[19] # Use spot price from a specific time
    strikes = [80, 90, 95, 100, 105, 110, 120]
    taus_days = [7, 14, 30, 90]
    taus_years = [d / 252.0 for d in taus_days]

    quotes = build_synthetic_quotes(S0, strikes, taus_years)
    surface = IVSurface.from_quotes(quotes, strikes=strikes, taus=taus_years)
    
    print("IV grid shape:", surface.iv_grid.shape)
    print("IV Grid:\n", np.round(surface.iv_grid, 4))

    # Query some IVs and compute greeks via OptionsEnv
    env = OptionsEnv(df, strike=100.0, expiry_index=350, option_type='call', window_size=10)
    # Attach surface to env instance
    env.iv_surface = surface
    
    # Reset env at the same time the quotes were generated for consistency
    obs, _ = env.reset(start_index=20) 
    
    print("\nImplied vol in obs (K=100):", obs.get('implied_vol'))
    print("Option delta:", obs.get('option_delta'))
    print("Option vega:", obs.get('option_vega'))
    print("Option theta:", obs.get('option_theta'))