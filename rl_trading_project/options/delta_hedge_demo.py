# Demo: run OptionsEnv with auto delta-hedging enabled and a simple policy that sells options and lets env hedge.
import numpy as np
import pandas as pd
from .options_env import OptionsEnv
from .iv_surface_demo import make_series

def run_demo():
    df = make_series(500, seed=3)
    env = OptionsEnv(df, strike=100.0, expiry_index=450, option_type='call', window_size=10, auto_hedge=True, commission=0.0001, slippage=0.0001)
    obs, _ = env.reset(start_index=20)
    total_reward = 0.0
    print(f"Initial State: total_value={obs['total_value']:.2f}")
    
    for t in range(50):
        # simple policy: sell small number of options each step, no manual hedge (env will auto-hedge)
        action = np.array([0.0, -0.02])  # keep underlying target 0, sell 2% of max_options
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += rew
        print(f"t={t}, reward={rew:.4f}, total_value={info['total_value']:.2f}, option_pos={info['option_pos']:.2f}, underlying_pos={info['underlying_pos']:.2f}, net_delta={info.get('net_delta', obs.get('net_delta')):.4f}")
        if done:
            print("Env done at step", t)
            break
            
    print("\nDemo finished. Total reward:", total_reward)
    print(f"Final State: total_value={info['total_value']:.2f}")

if __name__ == '__main__':
    run_demo()