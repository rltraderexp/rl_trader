# ppo_training_utils.py

import os
from rl_trading_project.envs import PortfolioEnv, GymWrapper

# This function is defined at the top level of a module, so it can be pickled
# and sent to child processes by multiprocessing.
def make_env_fn(df, window_size, seed):
    """A factory function to create a new environment instance."""
    def _fn():
        env = PortfolioEnv(
            df=df,
            window_size=window_size,
            initial_balance=100_000,
            max_leverage=2.0,
            commission=0.0005,
            reward_type='risk_adjusted'
        )
        
        # The old `env.seed(seed)` call is REMOVED from here.
        # Seeding is now handled by the Gymnasium vector environment's reset method.
        
        return GymWrapper(env)
    return _fn