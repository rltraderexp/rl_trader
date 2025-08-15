import pandas as pd
import numpy as np
from rl_trading_project.envs.simple_env import TradingEnv

def make_synthetic_df(n=200):
    rng = np.random.RandomState(42)
    price = 100 + np.cumsum(rng.normal(scale=0.2, size=n))
    dates = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price + rng.normal(scale=0.05, size=n),
        'high': price + 0.1,
        'low': price - 0.1,
        'close': price,
        'volume': rng.randint(1,100, size=n)
    })
    return df

def run_env_demo():
    df = make_synthetic_df(300)
    env = TradingEnv(df, window_size=10, initial_balance=10000, max_position=5.0, commission=0.001, slippage=0.001)
    
    obs, _ = env.reset(start_index=20)
    print("Initial total value:", env.total_value)
    
    # simple strategy: alternate between full long and full short
    total_rewards = 0.0
    for i in range(50):
        action = 1.0 if i % 2 == 0 else -1.0
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_rewards += reward
        if i < 5 or i > 45:
            env.render()
        if done:
            print("Environment finished early.")
            break
            
    print("\nTotal rewards:", total_rewards)
    print("Final total value:", env.total_value)

if __name__ == '__main__':
    run_env_demo()