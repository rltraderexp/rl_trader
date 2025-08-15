# Demo for the multi-asset PortfolioEnv
import numpy as np
import pandas as pd
from rl_trading_project.envs.portfolio_env import PortfolioEnv
from rl_trading_project.trainers.train_ppo import make_multi_asset_df

def run_portfolio_demo():
    df = make_multi_asset_df(n_assets=3, n_steps=200, seed=123)
    
    env = PortfolioEnv(df, window_size=10, initial_balance=100_000, max_leverage=2.0)
    
    obs, _ = env.reset(start_index=15)
    print("--- PortfolioEnv Demo ---")
    print(f"Assets: {env.assets}")
    print(f"Initial Total Value: {obs['total_value']:.2f}")
    
    # Simple policy: go long first asset, short second, neutral third
    action = np.array([0.8, -0.5, 0.0])
    
    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Step {i+1}:")
        print(f"  Total Value: {info['total_value']:.2f}, PnL: {info['pnl']:.2f}, Leverage: {info['leverage']:.2f}")
        print(f"  Positions: {np.round(info['positions'], 4)}")
        
        # Invert action every 3 steps
        if (i+1) % 3 == 0:
            action *= -1

        if done:
            print("\nEnvironment terminated.")
            break
            
    print("\nPortfolio demo finished.")

if __name__ == '__main__':
    run_portfolio_demo()