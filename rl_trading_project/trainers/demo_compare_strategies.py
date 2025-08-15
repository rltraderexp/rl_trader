# Demo: run two strategies and compare results, saving histories to CSV.
import numpy as np
from .backtester import Backtester, save_history_csv, compare_strategies
from ..options.iv_surface_demo import make_series
from ..options.options_env import OptionsEnv
import pandas as pd

def make_factory(auto_hedge, seed):
    def factory():
        df = make_series(600, seed=seed)
        return OptionsEnv(df, strike=100.0, expiry_index=550, option_type='call', auto_hedge=auto_hedge)
    return factory

# A simple policy that sells a small number of options
def sell_policy(obs,t):
    return np.array([0.0, -0.01]) # Target no underlying, sell 1% of max options

# A policy that does nothing
def hold_policy(obs,t):
    return np.array([0.0, 0.0])

if __name__ == '__main__':
    seed = 5
    policies = {
        'Sell_and_Hedge': (sell_policy, make_factory(True, seed)),
        'Sell_Unhedged': (sell_policy, make_factory(False, seed)),
        'Hold': (hold_policy, make_factory(False, seed))
    }
    
    results = {}
    for name, (policy, factory) in policies.items():
        bt = Backtester(factory, start_index=20)
        res = bt.run(policy, max_steps=300)
        results[name] = res
        save_history_csv(res['history'], f"{name}_history.csv")
        print(f"Finished backtest for: {name}")

    comparison = compare_strategies(results)
    
    print("\n--- Strategy Comparison ---")
    print(pd.DataFrame(comparison).T[['total_return', 'sharpe_ratio', 'max_drawdown']])