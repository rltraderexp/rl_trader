# Options env demo
import numpy as np
import pandas as pd
from .options_env import OptionsEnv

def make_series(n=200):
    rng = np.random.RandomState(42)
    price = 100 + np.cumsum(rng.normal(scale=0.5, size=n))
    dates = pd.date_range('2025-01-01', periods=n, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp':dates, 'open':price, 'high':price+0.1, 'low':price-0.1, 'close':price, 'volume':rng.randint(1,100,size=n)})
    return df

def run_demo():
    df = make_series(300)
    env = OptionsEnv(df, strike=105.0, expiry_index=250, option_type='call', window_size=10)
    obs, _ = env.reset(start_index=20)
    print('Initial option price:', obs['option_price'])
    
    for i in range(10):
        # random actions for demo
        action = [np.random.uniform(-1,1), np.random.uniform(-1,1)]
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Step {i}: total_value={info['total_value']:.2f}, pnl={info['pnl']:.2f}, option_pos={info['option_pos']:.2f}, underlying_pos={info['underlying_pos']:.2f}")
        if done:
            print("Environment finished early.")
            break

if __name__ == '__main__':
    run_demo()