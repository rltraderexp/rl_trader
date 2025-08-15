# Demo walk-forward backtest
import numpy as np
import pandas as pd
import os
from .walkforward import WalkForwardBacktester
from ..options.iv_surface_demo import make_series
from ..options.options_env import OptionsEnv

def env_factory_segment(df, start_idx, end_idx):
    def factory():
        return OptionsEnv(df, strike=100.0, expiry_index=end_idx-1, option_type='call')
    return factory

def sell_policy(obs, t):
    return np.array([0.0, -0.005])

def hold_policy(obs, t):
    return np.array([0.0, 0.0])

if __name__ == "__main__":
    # Create a single large dataset for the entire walk-forward process
    full_df = make_series(1200, seed=6)

    # The env_factory_fn now takes the full dataframe and returns a factory for a segment
    def get_env_factory(start, end):
        return env_factory_segment(full_df, start, end)

    wf = WalkForwardBacktester(env_factory_fn=get_env_factory, train_size=200, test_size=50, step=50)
    
    policies = {'sell': sell_policy, 'hold': hold_policy}
    
    save_directory = 'walkforward_out'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    res = wf.run(policies, start_index=20, end_index=600, max_folds=3, save_dir=save_directory)
    
    print("--- Walk-forward Mean Metrics ---")
    print(pd.DataFrame(res['mean_metrics']).T)