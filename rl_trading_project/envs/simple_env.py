"""
Simple trading environment (Gymnasium-like API) without external dependency.
"""
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    gym = object
    spaces = None
    GYM_AVAILABLE = False


class TradingEnv(gym.Env):
    """
    A simple single-asset trading environment that is compliant with the Gymnasium API.
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self,
                 df: pd.DataFrame,
                 window_size: int = 10,
                 initial_balance: float = 10_000.0,
                 max_position: float = 1.0,
                 commission: float = 0.0005,
                 slippage: float = 0.0005,
                 reward_type: str = "pnl"):
        
        super().__init__()
        
        assert 'timestamp' in df.columns and 'close' in df.columns, "DataFrame must contain timestamp and close"
        self.df = df.reset_index(drop=True)
        self.window_size = int(window_size)
        self.initial_balance = float(initial_balance)
        self.max_position = float(max_position)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.reward_type = reward_type
        self.render_mode = "human"

        # Internal state
        self._current_index = self.window_size
        self.position = 0.0
        self.cash = self.initial_balance

        # bookkeeping
        self.position_value = 0.0
        self.total_value = self.initial_balance
        
        # Define action and observation spaces required by gym.Env
        if GYM_AVAILABLE:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            
            # This is a placeholder for the wrapper's observation space.
            # The GymWrapper will compute the final, flattened observation space.
            # Assuming 5 features (o,h,l,c,v) + 3 portfolio states (pos, cash, price)
            obs_shape = self.window_size * 5 + 3
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        else:
            self.action_space = None
            self.observation_space = None


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict]:
        super().reset(seed=seed)
        start_index = options.get('start_index', self.window_size) if options else self.window_size

        if start_index < self.window_size or start_index >= len(self.df):
            raise ValueError("start_index must be in [window_size, len(df)-1]")
        
        self._current_index = int(start_index)
        self.position = 0.0
        self.cash = float(self.initial_balance)
        self.position_value = 0.0
        self.total_value = float(self.initial_balance)
        return self._get_obs(), {}

    def _get_obs(self) -> Dict[str, Any]:
        start = self._current_index - self.window_size
        window_df = self.df.iloc[start:self._current_index]
        cols = ['open','high','low','close','volume']
        available = [c for c in cols if c in window_df.columns]
        obs_window = window_df[available].to_numpy(dtype=np.float32)

        obs = {
            'window': obs_window,
            'position': np.float32(self.position),
            'cash': np.float32(self.cash),
            'current_price': np.float32(self.df.iloc[self._current_index]['close']),
            'timestamp': self.df.iloc[self._current_index]['timestamp']
        }
        return obs

    def step(self, action: float):
        terminated = False
        truncated = False
        
        action = float(action)
        if np.isnan(action):
            action = 0.0
        target_fraction = np.clip(action, -1.0, 1.0)
        target_pos = target_fraction * self.max_position

        price = float(self.df.iloc[self._current_index]['close'])
        trade = target_pos - self.position
        
        exec_price = price * (1.0 + np.sign(trade) * self.slippage)
        commission_cost = abs(trade * exec_price) * self.commission

        self.position += trade
        self.cash -= (trade * exec_price) + commission_cost

        self._current_index += 1
        if self._current_index >= len(self.df):
            self._current_index = len(self.df) - 1
            truncated = True

        mark_price = float(self.df.iloc[self._current_index]['close'])
        self.position_value = self.position * mark_price
        prev_total = self.total_value
        self.total_value = self.cash + self.position_value
        pnl = self.total_value - prev_total
        
        if self.total_value <= 0:
            terminated = True

        reward = float(pnl)

        info = {
            'position': self.position, 'cash': self.cash, 'mark_price': mark_price,
            'exec_price': exec_price, 'commission': commission_cost,
            'total_value': self.total_value, 'pnl': pnl, 'index': self._current_index
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print(f"Index: {self._current_index}, Time: {self.df.iloc[self._current_index]['timestamp']}, Price: {self.df.iloc[self._current_index]['close']:.4f}")
        print(f"Position: {self.position:.6f}, Cash: {self.cash:.2f}, Total: {self.total_value:.2f}")