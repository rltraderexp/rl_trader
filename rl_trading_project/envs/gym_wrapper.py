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


# =================================================================================
# --- START: MODIFIED GymWrapper CLASS ---
# =================================================================================

class GymWrapper(gym.Wrapper):
    def __init__(self, env, feature_fn: Optional[Callable]=None, forecast_adapter: Optional[Any]=None, forecast_horizon: int = 60):
        # Call the gym.Wrapper superclass constructor
        super().__init__(env)
        
        self.feature_fn = feature_fn
        self.forecast_adapter = forecast_adapter
        self.forecast_horizon = int(forecast_horizon)
        
        # The metadata dictionary is still needed for some checks.
        self.metadata = getattr(env, 'metadata', {'render_modes': []})
        
        # The line `self.render_mode = ...` has been REMOVED.
        # The superclass constructor already handles this.
        
        # Dynamically set the observation space based on the flattened vector
        if GYM_AVAILABLE:
            # Temporarily reset the env to get a sample observation for shape inference
            sample_obs_dict, _ = self.env.reset()
            sample_obs_vec = self._obs_to_vec(sample_obs_dict)
            
            self.observation_space = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=sample_obs_vec.shape, 
                dtype=np.float32
            )
            
            if self.env.action_space is None:
                if hasattr(self.env, 'n_assets'):
                    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.env.n_assets,), dtype=np.float32)
                else:
                    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.env.reset()
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        vec = self._obs_to_vec(obs)
        return vec, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Pass through important non-vector data from the original observation dict
        if isinstance(obs, dict):
            if 'timestamp' in obs:
                info['timestamp'] = obs['timestamp']
            if 'current_price' in obs:
                info['current_price'] = obs['current_price']

        vec = self._obs_to_vec(obs)
        # Pass through the 5 values expected by the Gymnasium API
        return vec, float(reward), bool(terminated), bool(truncated), info

    def _obs_to_vec(self, obs: Dict[str, Any]) -> np.ndarray:
        # Handle multi-asset (PortfolioEnv) vs single-asset (TradingEnv)
        if 'windows' in obs: # PortfolioEnv structure
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
            return base_vec
        
        else:
            raise ValueError("Observation dictionary structure not recognized.")

# =================================================================================
# --- END: MODIFIED GymWrapper CLASS ---
# =================================================================================


class SyncVectorEnv:
    def __init__(self, env_fns: List[callable]):
        assert len(env_fns) > 0
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        if self.num_envs > 0:
            self.observation_space = self.envs.observation_space
            self.action_space = self.envs.action_space
            self.obs_shape = self.observation_space.shape
        else:
            self.observation_space = None
            self.action_space = None
            self.obs_shape = None

    def reset(self, **kwargs):
        """Resets all environments and passes kwargs to each of them."""
        obs = [env.reset(**kwargs) for env in self.envs]
        return np.stack(obs, axis=0)

    def step(self, actions):
        rets = [env.step(action) for env, action in zip(self.envs, actions)]
        obs, rews, terminateds, truncateds, infos = zip(*rets)
        dones = np.logical_or(np.array(terminateds), np.array(truncateds))
        return np.stack(obs, axis=0), np.array(rews), np.array(dones), infos