# Demo for SyncVectorEnv with 4 parallel simple envs
import numpy as np
import pandas as pd
from rl_trading_project.envs.simple_env import TradingEnv
from rl_trading_project.envs.gym_wrapper import GymWrapper, SyncVectorEnv

def make_series(n=300):
    rng = np.random.RandomState(1)
    price = 100 + np.cumsum(rng.normal(scale=0.2, size=n))
    dates = pd.date_range('2025-01-01', periods=n, freq='min', tz='UTC')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price + rng.normal(scale=0.05, size=n),
        'high': price + 0.1,
        'low': price - 0.1,
        'close': price,
        'volume': rng.randint(1,100, size=n)
    })
    return df

def make_env_fn(seed=0):
    def _fn():
        df = make_series(500)
        env = TradingEnv(df, window_size=10, max_position=5.0)
        wrapped = GymWrapper(env)
        return wrapped
    return _fn

def run_vector_demo():
    num_envs = 4
    env_fns = [make_env_fn(i) for i in range(num_envs)]
    vec_env = SyncVectorEnv(env_fns)
    
    obs = vec_env.reset()
    print('Vectorized Env Demo')
    print('Observation shape:', obs.shape)
    assert obs.shape[0] == num_envs
    
    for i in range(5):
        actions = np.random.uniform(-1, 1, size=(num_envs, vec_env.action_space.shape[0]))
        obs, rewards, dones, infos = vec_env.step(actions)
        print(f"Step {i}: Rewards={np.round(rewards, 2)}, Dones={dones}")

if __name__ == "__main__":
    run_vector_demo()