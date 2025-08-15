"""
Orderbook-backed Trading Environment

Wraps the PortfolioEnv logic but uses an OrderBook simulator to execute trades.
This demonstrates replacing the naive slippage model with a market impact model.
NOTE: This implementation is simplified to work for a single asset.
"""
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from ..envs.portfolio_env import PortfolioEnv
from .orderbook import simulate_market_order

class OrderbookTradingEnv(PortfolioEnv):
    def __init__(self, df: pd.DataFrame, liquidity_depth: float = 1e5, impact_model: str = "sqrt", **kwargs):
        # This environment is intended for single-asset use cases to demonstrate execution models.
        # We ensure the parent is initialized for a single asset.
        if isinstance(df.index, pd.MultiIndex):
            assets = df.index.get_level_values('asset').unique()
            if len(assets) > 1:
                raise ValueError("OrderbookTradingEnv currently supports only single-asset DataFrames.")
        
        super().__init__(df, **kwargs)
        self.liquidity_depth = float(liquidity_depth)
        self.impact_model = impact_model

    def step(self, action: float):
        if self.done:
            raise RuntimeError("Step called on done environment. Call reset().")
        
        # Action for single asset
        action = float(action[0]) if hasattr(action, '__len__') else float(action)
        if np.isnan(action): action = 0.0
            
        current_price = self.df.loc[self.timestamps[self._current_ts_idx], 'close'].item()
        equity = self.cash + self.positions[0] * current_price
        
        target_leverage_frac = np.clip(action, -1.0, 1.0)
        target_pos = (target_leverage_frac * self.max_leverage * equity) / (current_price + 1e-12)
        trade = target_pos - self.positions[0]

        result = simulate_market_order(
            current_price, trade, self.liquidity_depth, self.commission, self.impact_model
        )
        filled, avg_price, commission_cost = result['filled_size'], result['avg_price'], result['commission']

        self.positions[0] += filled
        self.cash -= (filled * avg_price) + commission_cost
        
        funding_cost = (self.positions[0] * avg_price) * self.funding_rate
        self.cash -= funding_cost

        # Advance time and MTM (re-uses logic from parent class via its helpers)
        self._current_ts_idx += 1
        if self._current_ts_idx >= len(self.timestamps):
            self.done = True
            self._current_ts_idx = len(self.timestamps) - 1

        mark_price = self.df.loc[self.timestamps[self._current_ts_idx], 'close'].item()
        prev_total = self.total_value
        self.total_value = self.cash + self.positions[0] * mark_price
        pnl = self.total_value - prev_total
        
        self.peak_equity = max(self.peak_equity, self.total_value)
        self.drawdown = (self.peak_equity - self.total_value) / (self.peak_equity + 1e-12)

        notional = abs(self.positions[0] * mark_price)
        leverage = notional / (self.total_value + 1e-12)
        margin_called = False
        if (self.total_value <= 0) or (leverage / self.max_leverage > self.maintenance_margin_ratio):
            if self.positions[0] != 0:
                liq_result = simulate_market_order(mark_price, -self.positions[0], self.liquidity_depth, self.commission, self.impact_model)
                self.cash -= (liq_result['filled_size'] * liq_result['avg_price']) + liq_result['commission']
                self.positions[0] = 0
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
            'position': self.positions[0], 'cash': self.cash, 'mark_price': mark_price,
            'exec_price': avg_price, 'commission': commission_cost, 'funding_cost': funding_cost,
            'total_value': self.total_value, 'pnl': pnl, 'index': self._current_ts_idx,
            'margin_called': margin_called, 'leverage': leverage, 'drawdown': self.drawdown,
            'filled': filled
        }
        self.steps += 1
        return self._get_obs(), reward, self.done, False, info