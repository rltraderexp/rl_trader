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
    GYM_AVAILABLE = False

class PortfolioEnv:
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

        # Internal state
        self._current_ts_idx = None
        self.positions = np.zeros(self.n_assets, dtype=np.float32)
        self.cash = self.initial_balance
        self.done = False

        # Bookkeeping
        self.total_value = float(self.initial_balance)
        self.peak_equity = float(self.initial_balance)
        self.drawdown = 0.0
        self.steps = 0

        if GYM_AVAILABLE:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
            # Obs space is inferred by GymWrapper
            self.observation_space = None
        else:
            self.action_space = None
            self.observation_space = None

    def reset(self, start_index: Optional[int] = None) -> Tuple[Dict[str, Any], Dict]:
        if start_index is None:
            start_index = self.window_size
        if start_index < self.window_size or start_index >= len(self.timestamps):
            raise ValueError(f"start_index must be in [{self.window_size}, {len(self.timestamps)-1}]")
        
        self._current_ts_idx = int(start_index)
        self.positions = np.zeros(self.n_assets, dtype=np.float32)
        self.cash = float(self.initial_balance)
        self.total_value = float(self.initial_balance)
        self.peak_equity = float(self.initial_balance)
        self.drawdown = 0.0
        self.done = False
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self) -> Dict[str, Any]:
        current_ts = self.timestamps[self._current_ts_idx]
        start_ts = self.timestamps[self._current_ts_idx - self.window_size]
        
        # Efficiently slice the window for all assets
        window_df = self.df.loc[(slice(start_ts, current_ts), self.assets), self.obs_cols]
        window_vals = window_df.values.reshape(self.window_size + 1, self.n_assets, len(self.obs_cols))
        obs_window = window_vals[:-1, :, :].transpose(1, 0, 2) # (n_assets, window_size, n_features)

        current_prices = window_df.loc[current_ts].values[:, 3] # Close prices
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
        if self.done:
            # If the environment is already done, do not step further.
            # Return the last observation, a zero reward, and indicate termination.
            # This makes the environment compatible with fixed-length rollout collectors.
            return self._get_obs(), 0.0, True, False, {}
        
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        current_prices = self.df.loc[self.timestamps[self._current_ts_idx], 'close'].values
        equity = self.cash + (self.positions * current_prices).sum()

        target_leverage_fracs = actions
        target_notionals = target_leverage_fracs * self.max_leverage * equity / self.n_assets
        target_positions = target_notionals / (current_prices + 1e-12)
        trades = target_positions - self.positions
        
        exec_prices = current_prices * (1.0 + np.sign(trades) * self.slippage)
        commission_cost = np.abs(trades * exec_prices) * self.commission
        
        self.positions += trades
        self.cash -= (trades * exec_prices).sum() + commission_cost.sum()
        
        funding_cost = (self.positions * exec_prices) * self.funding_rate
        self.cash -= funding_cost.sum()
        
        self._current_ts_idx += 1
        if self._current_ts_idx >= len(self.timestamps):
            self.done = True
            self._current_ts_idx = len(self.timestamps) - 1

        mark_prices = self.df.loc[self.timestamps[self._current_ts_idx], 'close'].values
        prev_total = self.total_value
        position_values = self.positions * mark_prices
        self.total_value = self.cash + position_values.sum()
        pnl = self.total_value - prev_total

        self.peak_equity = max(self.peak_equity, self.total_value)
        self.drawdown = (self.peak_equity - self.total_value) / (self.peak_equity + 1e-12)

        notional = np.abs(position_values).sum()
        leverage = notional / (self.total_value + 1e-12)
        margin_called = False
        if (self.total_value <= 0) or (leverage / self.max_leverage > self.maintenance_margin_ratio):
            liq_comm = np.abs(self.positions * mark_prices) * self.commission
            self.cash += (self.positions * mark_prices).sum() - liq_comm.sum()
            self.positions.fill(0.0)
            self.total_value = self.cash
            self.done = True
            margin_called = True

        if self.reward_type == "pnl":
            reward = float(pnl)
        elif self.reward_type == "risk_adjusted":
            reward = float(pnl - self.drawdown * self.drawdown_penalty)
        else:
            reward = float(pnl)

        info = {
            'positions': self.positions,
            'cash': self.cash,
            'total_value': self.total_value,
            'pnl': pnl,
            'margin_called': margin_called,
            'leverage': leverage,
            'drawdown': self.drawdown
        }
        self.steps += 1
        return self._get_obs(), reward, self.done, False, info

    def render(self, mode='human'):
        print(f"Time: {self.timestamps[self._current_ts_idx]}, Total Value: {self.total_value:.2f}")
        print(f"Positions: {self.positions}, Cash: {self.cash:.2f}, Leverage: {self.total_value / (self.cash + 1e-9):.2f}")

    def seed(self, seed: Optional[int] = None):
        return seed