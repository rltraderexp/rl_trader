# Demo for OrderbookTradingEnv to show market impact behavior
import numpy as np
import pandas as pd
from rl_trading_project.execution.orderbook_env import OrderbookTradingEnv

def make_price_series(n=300):
    rng = np.random.RandomState(2025)
    price = 100 + np.cumsum(rng.normal(scale=0.5, size=n))
    # Fix FutureWarning: 'T' is deprecated, use 'min' for minutes
    dates = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    df = pd.DataFrame({
        'timestamp': dates, 'close': price, 'asset': 'ASSET_0',
        'open': price + rng.normal(scale=0.1, size=n),
        'high': price + 0.2, 'low': price - 0.2,
        'volume': rng.randint(100,1000, size=n)
    }).set_index(['timestamp', 'asset'])
    return df

def run_impact_demo():
    df = make_price_series(200)
    # Increase liquidity depth to make impact visible but not instantly fatal
    env = OrderbookTradingEnv(df, window_size=10, initial_balance=100000.0, max_leverage=2.0,
                              liquidity_depth=5e5, impact_model='sqrt') # Increased from 5e4 to 5e5
    obs, _ = env.reset(start_index=20)
    
    mark_price_before = obs['current_prices'][0]
    print(f"Initial Equity: {obs['total_value']:.2f}, Mark Price: {mark_price_before:.4f}")

    # Place a large buy order (target full long) to observe positive slippage
    obs, reward, terminated, truncated, info = env.step(np.array([1.0]))
    done = terminated or truncated
    print(f"\nAfter large buy:")
    print(f"  Execution Price: {info['exec_price']:.4f} (slippage: {info['exec_price'] - mark_price_before:.4f})")
    print(f"  Filled: {info['filled']:.2f}, Total Value: {info['total_value']:.2f}")

    # CRITICAL FIX: Check if the environment is done before stepping again
    if done:
        print("\nEnvironment terminated after the first trade (e.g., due to margin call). Demo finished.")
        return

    # Place a large sell order (target full short) to observe negative slippage
    mark_price_before = obs['current_prices'][0]
    obs, reward, terminated, truncated, info = env.step(np.array([-1.0]))
    print(f"\nAfter large sell:")
    print(f"  Execution Price: {info['exec_price']:.4f} (slippage: {info['exec_price'] - mark_price_before:.4f})")
    print(f"  Filled: {info['filled']:.2f}, Total Value: {info['total_value']:.2f}")

if __name__ == '__main__':
    run_impact_demo()