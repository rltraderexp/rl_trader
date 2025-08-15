# Demo backtest: compare auto-hedged vs unhedged option selling policy using Backtester.
import numpy as np
import pandas as pd
from ..options.options_env import OptionsEnv
from ..options.iv_surface_demo import make_series
from .backtester import Backtester, compare_strategies

def make_env_factory(auto_hedge: bool, seed: int):
    def factory():
        df = make_series(1000, seed=seed)
        return OptionsEnv(df, strike=100.0, expiry_index=900, option_type='call', 
                          auto_hedge=auto_hedge, commission=0.0001, slippage=0.0001)
    return factory

def sell_options_policy(obs, t):
    # simple policy: sell 1 option contract per step (as a proportion of max_options)
    return np.array([0.0, -0.005])

def run_backtests():
    seed = 4
    
    # Hedged strategy
    factory_hedged = make_env_factory(auto_hedge=True, seed=seed)
    bt_hedged = Backtester(factory_hedged, start_index=20)
    res_hedged = bt_hedged.run(sell_options_policy, max_steps=500)
    
    # Unhedged strategy
    factory_unhedged = make_env_factory(auto_hedge=False, seed=seed)
    bt_unhedged = Backtester(factory_unhedged, start_index=20)
    res_unhedged = bt_unhedged.run(sell_options_policy, max_steps=500)
    
    # Compare results
    comparison = compare_strategies({
        'Hedged': res_hedged,
        'Unhedged': res_unhedged
    })
    
    print("--- Comparison Summary ---")
    print(pd.DataFrame(comparison).T[['total_return', 'sharpe_ratio', 'max_drawdown']])

if __name__ == '__main__':
    run_backtests()