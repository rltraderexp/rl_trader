# Simple training loop demo for DuelingDQNAgent with the simple TradingEnv
import numpy as np
import pandas as pd
from ..agents.dqn import DuelingDQNAgent
from ..envs.simple_env import TradingEnv
from ..envs.gym_wrapper import GymWrapper

def make_series(n=500):
    rng = np.random.RandomState(0)
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

def train_demo(steps=3000):
    df = make_series(1000)
    base_env = TradingEnv(df, window_size=10, initial_balance=10000, max_position=5.0, commission=0.001, slippage=0.0005)
    env = GymWrapper(base_env) # Use GymWrapper for observation flattening
    
    obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    
    agent = DuelingDQNAgent(
        obs_dim, 
        action_bins=11, 
        hidden=(64,64), 
        lr=1e-3, 
        buffer_size=50000, 
        batch_size=64,
        device='cpu'
    )
    
    total_reward = 0.0
    for step in range(steps):
        action = agent.act(obs, deterministic=False)
        next_obs, reward, done, info = env.step(action)
        
        agent.add_experience(obs, action, reward, next_obs, done)
        
        stats = agent.update(sync_freq=100)
        
        obs = next_obs
        if done:
            # For simplicity in demo, we just reset the base env. A more robust setup would handle this differently.
            obs, _ = env.reset()

        total_reward += reward if reward is not None else 0.0
        
        if step > 0 and step % 200 == 0:
            print(f"Step {step}, Loss={stats.get('loss'):.4f}, Epsilon={stats.get('eps'):.2f}, Beta={stats.get('beta'):.2f}, Total Reward={total_reward:.2f}")
            total_reward = 0.0 # Reset episodic reward counter for clarity
            
    print('\nTraining demo finished.')
    return agent

if __name__ == '__main__':
    train_demo(steps=5000)