"""
GymWrapper with feature and forecast integration.

Dynamically determines observation shape from the environment and flattens
complex observations (e.g., from PortfolioEnv) into a single vector for the agent.
"""
from typing import List, Any, Dict, Optional, Callable
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    gym = None
    spaces = None
    GYM_AVAILABLE = False

class GymWrapper:
    def __init__(self, env, feature_fn: Optional[Callable]=None, forecast_adapter: Optional[Any]=None, forecast_horizon: int = 60):
        self.env = env
        self.feature_fn = feature_fn
        self.forecast_adapter = forecast_adapter
        self.forecast_horizon = int(forecast_horizon)
        
        # Dynamically set observation and action spaces from env
        if GYM_AVAILABLE:
            sample_obs = self._obs_to_vec(self.env.reset()[0])
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32)
            
            # Infer action space from env if possible, otherwise default
            if hasattr(self.env, 'action_space') and self.env.action_space is not None:
                self.action_space = self.env.action_space
            elif hasattr(self.env, 'n_assets'):
                 self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.env.n_assets,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.observation_space = None
            self.action_space = None
        
        # Reset again to ensure env state is clean for the user
        self.env.reset()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        vec = self._obs_to_vec(obs)
        return vec, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        vec = self._obs_to_vec(obs)
        # Pass through the 5 values expected by the Gymnasium API and SyncVectorEnv
        return vec, float(reward), bool(terminated), bool(truncated), info

    def _obs_to_vec(self, obs: Dict[str, Any]) -> np.ndarray:
        # Handle multi-asset (PortfolioEnv) vs single-asset (TradingEnv)
        if 'windows' in obs: # PortfolioEnv structure
            # obs['windows'] is (n_assets, window_size, n_features)
            market_state = obs['windows'].flatten()
            portfolio_state = np.concatenate([
                obs['positions'],
                [obs['cash'], obs['total_value'], obs['leverage']]
            ]).astype(np.float32)
            return np.concatenate([market_state, portfolio_state])
        
        elif 'window' in obs: # simple_env or other single-asset envs
            w = obs['window'].astype(np.float32).flatten()
            extras = [obs['position'], obs['cash'], obs['current_price']]
            base_vec = np.concatenate([w, np.array(extras, dtype=np.float32)])
            # NOTE: feature_fn and forecast_adapter are currently only supported for single-asset envs
            # for simplicity. Extending them to multi-asset would require careful API design.
            return base_vec
        
        else:
            raise ValueError("Observation dictionary structure not recognized.")


class SyncVectorEnv:
    def __init__(self, env_fns: List[callable]):
        assert len(env_fns) > 0
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        if self.num_envs > 0:
            self.observation_space = self.envs[0].observation_space
            self.action_space = self.envs[0].action_space
            self.obs_shape = self.observation_space.shape
        else: # Should not happen with assertion but for safety
            self.observation_space = None
            self.action_space = None
            self.obs_shape = None


    def reset(self):
        obs = [env.reset()[0] for env in self.envs]
        return np.stack(obs, axis=0)

    def step(self, actions):
        rets = [env.step(action) for env, action in zip(self.envs, actions)]
        obs, rews, terminateds, truncateds, infos = zip(*rets)
        dones = np.logical_or(np.array(terminateds), np.array(truncateds))
        return np.stack(obs, axis=0), np.array(rews), np.array(dones), infos