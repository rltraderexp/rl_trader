"""
PPO training harness for the multi-asset PortfolioEnv.

This script demonstrates training a PPO agent on a portfolio of assets,
integrating feature engineering and robust scaling of observations.
"""
import numpy as np
import pandas as pd
import random
import torch
from ..agents.ppo import PPOAgent
from ..envs.gym_wrapper import GymWrapper, SyncVectorEnv
from ..envs.portfolio_env import PortfolioEnv
from ..features.basic_features import assemble_features
from ..preprocessing.scaler import ScalerWrapper
from ..utils.experiments import ExperimentLogger

def make_multi_asset_df(n_assets=3, n_steps=1000, seed=0):
    """Generates a multi-indexed DataFrame for multiple assets."""
    rng = np.random.RandomState(seed)
    all_dfs = []
    assets = [f'ASSET_{i}' for i in range(n_assets)]
    
    for i, asset in enumerate(assets):
        # Create correlated but distinct price series
        base_drift = rng.normal(0.0001, 0.00005)
        base_vol = rng.uniform(0.1, 0.3)
        
        price = 100 * (1 + rng.uniform(-0.2, 0.2)) + np.cumsum(rng.normal(base_drift, base_vol, size=n_steps))
        
        dates = pd.date_range('2025-01-01', periods=n_steps, freq='min', tz='UTC')
        df = pd.DataFrame({
            'timestamp': dates,
            'asset': asset,
            'open': price + rng.normal(0, 0.05, size=n_steps),
            'high': price + rng.uniform(0, 0.1, size=n_steps),
            'low': price - rng.uniform(0, 0.1, size=n_steps),
            'close': price,
            'volume': rng.randint(100, 10000, size=n_steps)
        })
        all_dfs.append(df)
        
    full_df = pd.concat(all_dfs).set_index(['timestamp', 'asset']).sort_index()
    return full_df

def make_env_fn(df, scaler, seed=0):
    def _fn():
        # Feature engineering: apply to the whole df, then the env will slice it.
        # This is more efficient than re-calculating on each step.
        # The scaler is applied to the engineered features.
        
        # FIX: Added group_keys=False to prevent duplicate 'asset' level in the index.
        featured_df = df.groupby(level='asset', group_keys=False).apply(assemble_features).pipe(scaler.transform_df)
        
        env = PortfolioEnv(featured_df, window_size=15, max_leverage=2.0, reward_type='risk_adjusted')
        env.seed(seed)
        wrapped_env = GymWrapper(env)
        return wrapped_env
    return _fn

def evaluate_agent(agent, env_fn, episodes=5, max_steps=500):
    """Run deterministic evaluation episodes and return average total return."""
    returns = []
    for _ in range(episodes):
        env = env_fn()
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < max_steps:
            action, _, _ = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        returns.append(total_reward)
    return {'avg_return': np.mean(returns)}

def run_experiment(exp_name='ppo_portfolio_exp', seed=42, n_assets=3, num_envs=4, total_steps=10000, config=None):
    if config is None: config = {}
    
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    logger = ExperimentLogger(base_dir='runs')
    logger.log_metrics({'experiment': exp_name, 'seed': seed, 'n_assets': n_assets}, step=0)

    # 1. Create and preprocess data
    df_raw = make_multi_asset_df(n_assets=n_assets, n_steps=2000, seed=seed)
    
    # 2. Fit scaler on the first half of the data (training period)
    scaler = ScalerWrapper(method='standard')
    # Fit on features of the training slice
    train_slice_df = df_raw.groupby(level='asset', group_keys=False).apply(assemble_features).iloc[:1000 * n_assets]
    scaler.fit(train_slice_df)

    # 3. Create vectorized environments
    env_fns = [make_env_fn(df_raw, scaler, seed=seed + i) for i in range(num_envs)]
    vec_env = SyncVectorEnv(env_fns)
    
    agent = PPOAgent(
        obs_dim=vec_env.obs_shape[0],
        action_dim=vec_env.action_space.shape[0],
        seed=seed,
        device='cpu',
        **config.get('agent_params', {})
    )

    rollout_len = config.get('rollout_len', 256)
    eval_every = config.get('eval_every', 2048)
    
    obs, trajectories = vec_env.reset(), []
    steps_done, best_eval_return = 0, -np.inf
    
    while steps_done < total_steps:
        for _ in range(rollout_len):
            actions_for_step, partial_transitions = [], []
            for env_idx in range(num_envs):
                action, logp, value = agent.act(obs[env_idx], deterministic=False)
                actions_for_step.append(action)
                partial_transitions.append({'obs': obs[env_idx], 'act': action, 'logp': logp, 'value': value})
            
            next_obs, rewards, dones, _ = vec_env.step(actions_for_step)
            
            for i in range(num_envs):
                transition = partial_transitions[i]
                transition.update({'rew': rewards[i], 'next_obs': next_obs[i], 'done': dones[i]})
                trajectories.append(transition)

            obs = next_obs
            steps_done += num_envs

        stats = agent.update(trajectories)
        trajectories.clear()
        logger.log_metrics(stats, step=steps_done)
        print(f"[{exp_name}] Step: {steps_done}/{total_steps}, Stats: {stats}")

        if steps_done % eval_every == 0:
            eval_results = evaluate_agent(agent, make_env_fn(df_raw, scaler, seed=seed + 999), episodes=5)
            logger.log_metrics({'eval_avg_return': eval_results['avg_return']}, step=steps_done)
            print(f"[{exp_name}] Eval @ step {steps_done}: Avg Return = {eval_results['avg_return']:.2f}")

            if eval_results['avg_return'] > best_eval_return:
                best_eval_return = eval_results['avg_return']
                logger.save_checkpoint(agent, name=f'{exp_name}_best.pth')

    logger.save_checkpoint(agent, name=f'{exp_name}_final.pth')
    print("Training finished.")
    return agent, logger

if __name__ == '__main__':
    run_experiment(total_steps=20_000)