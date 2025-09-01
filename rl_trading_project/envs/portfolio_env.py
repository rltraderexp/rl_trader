"""
Multi-Asset Portfolio Trading Environment with leverage, margin, and risk-aware rewards.

This environment simulates managing a portfolio of multiple assets.
- Data `df` must be a pandas DataFrame with a MultiIndex `(timestamp, asset)`.
- Action is a vector of target leverage fractions, one for each asset in `[-1,1]`.
- State includes historical windows for all assets and current portfolio state (positions, cash).
- Simulates commission, slippage, funding costs, and margin calls (liquidation).
"""
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    gym = object  # Define a fallback if gym is not available
    spaces = None
    GYM_AVAILABLE = False


class PortfolioEnv(gym.Env):
    """
    A multi-asset portfolio management environment that is compliant with the Gymnasium API.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 df: pd.DataFrame,
                 window_size: int = 10,
                 initial_balance: float = 10_000.0,
                 max_leverage: float = 2.0,
                 commission: float = 0.0005,
                 slippage: float = 0.0005,
                 funding_rate: float = 0.0,
                 maintenance_margin_ratio: float = 0.8,
                 reward_type: str = "pnl",
                 drawdown_penalty: float = 100.0):
        
        super().__init__()
        
        assert isinstance(df.index, pd.MultiIndex), "df must have a MultiIndex (timestamp, asset)"
        self.df = df.sort_index()
        self.timestamps = self.df.index.get_level_values('timestamp').unique().sort_values()
        self.assets = self.df.index.get_level_values('asset').unique().tolist()
        self.n_assets = len(self.assets)
        self.obs_cols = ['open', 'high', 'low', 'close', 'volume']
        
        self.window_size = int(window_size)
        self.initial_balance = float(initial_balance)
        self.max_leverage = float(max_leverage)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.funding_rate = float(funding_rate)
        self.maintenance_margin_ratio = float(maintenance_margin_ratio)
        self.reward_type = reward_type
        self.drawdown_penalty = float(drawdown_penalty)
        self.render_mode = "human"

        # Internal state
        self._current_ts_idx = 0
        self.positions = np.zeros(self.n_assets, dtype=np.float32)
        self.cash = self.initial_balance
        
        # Bookkeeping
        self.total_value = float(self.initial_balance)
        self.peak_equity = float(self.initial_balance)
        self.drawdown = 0.0
        self.steps = 0

        # Define action and observation spaces required by gym.Env
        if GYM_AVAILABLE:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
            
            # This is a placeholder for the wrapper's observation space.
            # The GymWrapper will compute the final, flattened observation space.
            market_state_shape = self.n_assets * self.window_size * len(self.obs_cols)
            portfolio_state_shape = self.n_assets + 3 # positions, cash, total_value, leverage
            total_obs_shape = market_state_shape + portfolio_state_shape
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_shape,), dtype=np.float32)
        else:
            self.action_space = None
            self.observation_space = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict]:
        super().reset(seed=seed)
        start_index = options.get('start_index', self.window_size) if options else self.window_size
        
        if start_index < self.window_size or start_index >= len(self.timestamps):
            raise ValueError(f"start_index must be in [{self.window_size}, {len(self.timestamps)-1}]")
        
        self._current_ts_idx = int(start_index)
        self.positions = np.zeros(self.n_assets, dtype=np.float32)
        self.cash = float(self.initial_balance)
        self.total_value = float(self.initial_balance)
        self.peak_equity = float(self.initial_balance)
        self.drawdown = 0.0
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self) -> Dict[str, Any]:
        start_idx = self._current_ts_idx - self.window_size
        end_idx = self._current_ts_idx + 1
        expected_timestamps = self.timestamps[start_idx:end_idx]
        current_ts = expected_timestamps[-1]
        
        all_asset_windows = []
        current_prices_list = []

        for asset in self.assets:
            asset_df = self.df.xs(asset, level='asset', drop_level=True)
            asset_window_df = asset_df.reindex(expected_timestamps, method='ffill').fillna(0.0)
            all_asset_windows.append(asset_window_df[self.obs_cols].values)
            current_prices_list.append(asset_window_df['close'].iloc[-1])

        full_window_vals = np.stack(all_asset_windows, axis=0)
        obs_window = full_window_vals[:, :-1, :]
        current_prices = np.array(current_prices_list, dtype=np.float32)

        equity = self.cash + (self.positions * current_prices).sum()
        notional_value = np.abs(self.positions * current_prices).sum()
        leverage = notional_value / (equity + 1e-12)
        
        obs = {
            'windows': obs_window.astype(np.float32),
            'positions': self.positions.astype(np.float32),
            'cash': np.float32(self.cash),
            'total_value': np.float32(equity),
            'leverage': np.float32(leverage),
            'peak_equity': np.float32(self.peak_equity),
            'drawdown': np.float32(self.drawdown),
            'current_prices': current_prices.astype(np.float32),
            'timestamp': current_ts
        }
        return obs

    def step(self, actions: np.ndarray):
        terminated = False
        truncated = False
        
        if self._current_ts_idx >= len(self.timestamps) -1:
            truncated = True
            return self._get_obs(), 0.0, terminated, truncated, {}

        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        obs_data = self._get_obs()
        current_prices = obs_data['current_prices']
        equity = obs_data['total_value']

        target_leverage_fracs = actions
        safe_prices = current_prices + 1e-12
        target_notionals = target_leverage_fracs * self.max_leverage * equity / self.n_assets
        target_positions = target_notionals / safe_prices
        trades = target_positions - self.positions
        
        exec_prices = current_prices * (1.0 + np.sign(trades) * self.slippage)
        commission_cost = np.abs(trades * exec_prices) * self.commission
        
        self.positions += trades
        self.cash -= (trades * exec_prices).sum() + commission_cost.sum()
        funding_cost = (self.positions * exec_prices) * self.funding_rate
        self.cash -= funding_cost.sum()
        
        self._current_ts_idx += 1
        if self._current_ts_idx >= len(self.timestamps) -1:
            truncated = True

        next_obs_data = self._get_obs()
        prev_total = self.total_value
        self.total_value = next_obs_data['total_value']
        pnl = self.total_value - prev_total

        self.peak_equity = max(self.peak_equity, self.total_value)
        self.drawdown = (self.peak_equity - self.total_value) / (self.peak_equity + 1e-12)

        margin_called = False
        if self.total_value <= 0 or next_obs_data['leverage'] / self.max_leverage > self.maintenance_margin_ratio:
            liq_comm = np.abs(self.positions * next_obs_data['current_prices']) * self.commission
            self.cash += (self.positions * next_obs_data['current_prices']).sum() - liq_comm.sum()
            self.positions.fill(0.0)
            self.total_value = self.cash
            terminated = True
            margin_called = True

        reward = float(pnl)
        if self.reward_type == "risk_adjusted":
            reward -= float(self.drawdown * self.drawdown_penalty)

        info = {
            'positions': self.positions.copy(), 'cash': self.cash, 'total_value': self.total_value,
            'pnl': pnl, 'margin_called': margin_called, 'leverage': next_obs_data['leverage'],
            'drawdown': self.drawdown
        }
        self.steps += 1
        return next_obs_data, reward, terminated, truncated, info

    def render(self):
        print(f"Time: {self.timestamps[self._current_ts_idx]}, Total Value: {self.total_value:.2f}")
        print(f"Positions: {self.positions}, Cash: {self.cash:.2f}, Leverage: {self.total_value / (self.cash + 1e-9):.2f}")