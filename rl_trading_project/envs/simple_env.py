"""
Simple trading environment (Gymnasium-like API) without external dependency.

Provides a minimal, deterministic environment suitable for unit tests and
fast experiments. Designed to accept canonicalized OHLCV DataFrame
(rows sorted by timestamp) and simulate trading with slippage & commission.

API:
    env = TradingEnv(df, window_size=10, initial_balance=10_000, max_position=1.0,
                     commission=0.0005, slippage=0.0005)
    obs, info = env.reset(start_index=100)
    obs, reward, terminated, truncated, info = env.step(action)  # action in [-1, 1] target position fraction
    env.render()
"""
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

class TradingEnv:
    def __init__(self,
                 df: pd.DataFrame,
                 window_size: int = 10,
                 initial_balance: float = 10_000.0,
                 max_position: float = 1.0,
                 commission: float = 0.0005,
                 slippage: float = 0.0005,
                 reward_type: str = "pnl"):
        assert 'timestamp' in df.columns and 'close' in df.columns, "DataFrame must contain timestamp and close"
        self.df = df.reset_index(drop=True)
        self.window_size = int(window_size)
        self.initial_balance = float(initial_balance)
        self.max_position = float(max_position)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.reward_type = reward_type

        # Internal state
        self._start_index = self.window_size
        self._current_index = None
        self.position = 0.0  # number of units (can be fractional, positive long, negative short)
        self.cash = self.initial_balance
        self.done = False

        # bookkeeping
        self.position_value = 0.0
        self.total_value = self.initial_balance

    def reset(self, start_index: Optional[int] = None) -> Tuple[Dict[str, Any], Dict]:
        """Reset environment. If start_index is None picks window_size as start."""
        if start_index is None:
            start_index = self.window_size
        if start_index < self.window_size or start_index >= len(self.df):
            raise ValueError("start_index must be in [window_size, len(df)-1]")
        self._start_index = int(start_index)
        self._current_index = int(start_index)
        self.position = 0.0
        self.cash = float(self.initial_balance)
        self.position_value = 0.0
        self.total_value = float(self.initial_balance)
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self) -> Dict[str, Any]:
        """Return observation dict with historical window and portfolio state."""
        start = self._current_index - self.window_size
        window_df = self.df.iloc[start:self._current_index].copy()
        # features: open, high, low, close, volume, vwap (if present)
        cols = ['open','high','low','close','volume']
        available = [c for c in cols if c in window_df.columns]
        obs_window = window_df[available].to_numpy(dtype=np.float32)
        # pad if needed (shouldn't for normal resets)
        if obs_window.shape[0] < self.window_size:
            pad = np.zeros((self.window_size - obs_window.shape[0], obs_window.shape[1]), dtype=np.float32)
            obs_window = np.vstack([pad, obs_window])
        obs = {
            'window': obs_window,          # shape (window_size, n_features)
            'position': np.float32(self.position),
            'cash': np.float32(self.cash),
            'current_price': np.float32(self.df.iloc[self._current_index]['close']),
            'timestamp': self.df.iloc[self._current_index]['timestamp']
        }
        return obs

    def step(self, action: float):
        """
        action: target position fraction in [-1,1] where 1 means long `max_position` units,
                -1 means short `max_position` units.
        Returns: obs, reward, terminated, truncated, info
        """
        if self.done:
            raise RuntimeError("Step called on done environment. Call reset().")
        # clip action
        action = float(action)
        if np.isnan(action):
            action = 0.0
        target_fraction = np.clip(action, -1.0, 1.0)
        target_pos = target_fraction * self.max_position

        # current market price (we assume immediate execution at next tick's close price)
        price = float(self.df.iloc[self._current_index]['close'])
        # compute trade
        trade = target_pos - self.position
        
        # simulate slippage: execution price moves against direction of trade
        exec_price = price
        if trade != 0.0:
            direction = np.sign(trade)
            exec_price = price * (1.0 + direction * self.slippage)

        # commission based on traded notional
        commission_cost = abs(trade * exec_price) * self.commission

        # Update cash and position (we assume unlimited margin for simplicity; can extend later)
        self.position += trade
        self.cash -= (trade * exec_price) + commission_cost

        # Advance time by one step (mark-to-market)
        self._current_index += 1
        if self._current_index >= len(self.df):
            self.done = True
            self._current_index = len(self.df) - 1

        # update valuation
        mark_price = float(self.df.iloc[self._current_index]['close'])
        self.position_value = self.position * mark_price
        prev_total = self.total_value
        self.total_value = self.cash + self.position_value
        pnl = self.total_value - prev_total

        # reward
        if self.reward_type == "pnl":
            reward = float(pnl)
        elif self.reward_type == "sharpe":  # simplistic: reward scaled by small factor
            # Note: This is a very basic form of Sharpe, proper implementation would track returns over time.
            reward = float(pnl) / (abs(self.position_value) + 1e-6) if self.position_value != 0 else float(pnl)
        else:
            reward = float(pnl)

        info = {
            'position': self.position,
            'cash': self.cash,
            'mark_price': mark_price,
            'exec_price': exec_price,
            'commission': commission_cost,
            'total_value': self.total_value,
            'pnl': pnl,
            'index': self._current_index
        }
        return self._get_obs(), reward, self.done, False, info

    def render(self, mode='human'):
        print(f"Index: {self._current_index}, Time: {self.df.iloc[self._current_index]['timestamp']}, Price: {self.df.iloc[self._current_index]['close']:.4f}")
        print(f"Position: {self.position:.6f}, Cash: {self.cash:.2f}, Total: {self.total_value:.2f}")

    def seed(self, seed: Optional[int] = None):
        # deterministic for now
        return seed