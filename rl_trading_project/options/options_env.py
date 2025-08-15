"""Options trading environment with delta-hedging primitives and simple margin accounting.

Agent action (2-dim numpy array):
    action[0] in [-1,1] -> target underlying position proportion of max_underlying (signed).
    action[1] in [-1,1] -> option trade amount proportion of max_options (signed): positive = buy options, negative = sell.
If `auto_hedge=True` (env config), the env will perform automatic delta-hedging after agent's action.

Observation includes market window, option greeks, positions, cash, and portfolio-level greeks.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from .black_scholes import bs_price, bs_vega, bs_delta, bs_theta

class OptionsEnv:
    def __init__(self, df: pd.DataFrame, strike: float, expiry_index: int, option_type: str = 'call',
                 window_size: int = 10, initial_balance: float = 100000.0,
                 max_underlying: float = 100.0, max_options: float = 1000.0,
                 commission: float = 0.0, slippage: float = 0.0, auto_hedge: bool = False,
                 margin_ratio: float = 0.2, maintenance_margin: float = 0.1,
                 reward_shaper: Optional[Any] = None, risk_aversion: float = 0.1):
        assert 'timestamp' in df.columns and 'close' in df.columns
        self.df = df.reset_index(drop=True)
        self.window_size = int(window_size)
        self.strike = float(strike)
        self.expiry_index = int(expiry_index)
        self.option_type = option_type
        self.initial_balance = float(initial_balance)
        self.max_underlying = float(max_underlying)
        self.max_options = float(max_options)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.auto_hedge = bool(auto_hedge)
        self.margin_ratio = float(margin_ratio)
        self.maintenance_margin = float(maintenance_margin)
        self.reward_shaper = reward_shaper
        self.risk_aversion = risk_aversion
        self.iv_surface = None # Can be set externally

        self.reset()

    def reset(self, start_index: Optional[int] = None) -> Tuple[Dict[str, Any], Dict]:
        if start_index is None:
            start_index = self.window_size
        self._current_index = int(start_index)
        self.underlying_pos = 0.0
        self.option_pos = 0.0
        self.cash = float(self.initial_balance)
        self.done = False
        self.total_value = float(self.initial_balance)
        self.info = {}
        self.value_history = [self.initial_balance]
        
        return self._get_obs(), {}

    def _get_obs(self) -> Dict[str, Any]:
        start = self._current_index - self.window_size
        window = self.df.iloc[start:self._current_index]
        price = float(self.df.iloc[self._current_index]['close'])
        
        minutes_to_expiry = max(0, (self.expiry_index - self._current_index))
        tau = max(1e-9, minutes_to_expiry / (252.0 * 390.0)) # 390 mins in a trading day
        
        iv = 0.2 # Default IV
        if self.iv_surface:
            iv = self.iv_surface.interp_iv(self.strike, tau)
        
        option_price = bs_price(price, self.strike, 0.0, iv, tau, self.option_type)
        delta = bs_delta(price, self.strike, 0.0, iv, tau, self.option_type)
        vega = bs_vega(price, self.strike, 0.0, iv, tau)
        theta = bs_theta(price, self.strike, 0.0, iv, tau, self.option_type)

        net_delta = delta * self.option_pos + self.underlying_pos
        net_vega = vega * self.option_pos
        
        total_value = self.cash + self.underlying_pos * price + self.option_pos * option_price
        
        option_notional = abs(self.option_pos) * option_price
        margin_req = self.margin_ratio * option_notional + self.maintenance_margin * abs(self.underlying_pos) * price
        
        obs = {
            'window': window[['open','high','low','close','volume']].to_numpy(dtype=np.float32),
            'price': price,
            'option_price': option_price,
            'implied_vol': float(iv),
            'option_delta': float(delta),
            'option_vega': float(vega),
            'option_theta': float(theta),
            'underlying_pos': float(self.underlying_pos),
            'option_pos': float(self.option_pos),
            'cash': float(self.cash),
            'net_delta': float(net_delta),
            'net_vega': float(net_vega),
            'margin_requirement': float(margin_req),
            'total_value': float(total_value),
            'tau': tau,
            'timestamp': self.df.iloc[self._current_index]['timestamp']
        }
        return obs

    def _execute_trade(self, pos_var, max_pos, trade_units, price):
        current_pos = getattr(self, pos_var)
        target = np.clip(current_pos + trade_units, -max_pos, max_pos)
        actual_trade = target - current_pos

        slippage_cost = abs(actual_trade) * self.slippage * price
        commission_cost = abs(actual_trade) * price * self.commission
        cost = actual_trade * price + slippage_cost + commission_cost
        
        setattr(self, pos_var, current_pos + actual_trade)
        self.cash -= cost
        return cost

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.done:
            return self._get_obs(), 0.0, self.done, False, self.info
            
        action = np.asarray(action, dtype=float).flatten()
        target_u_prop = float(np.clip(action[0], -1.0, 1.0))
        trade_o_prop = float(np.clip(action[1], -1.0, 1.0))
        
        obs_before_trade = self._get_obs()
        price = obs_before_trade['price']
        option_price = obs_before_trade['option_price']
        option_delta = obs_before_trade['option_delta']

        # Execute trades
        trade_o = trade_o_prop * self.max_options
        cost_o = self._execute_trade('option_pos', self.max_options, trade_o, option_price)
        
        target_u = target_u_prop * self.max_underlying
        trade_u = target_u - self.underlying_pos
        cost_u = self._execute_trade('underlying_pos', self.max_underlying, trade_u, price)
        
        cost_auto_hedge, auto_hedge_trade = 0.0, 0.0
        if self.auto_hedge:
            net_delta_after_trade = option_delta * self.option_pos + self.underlying_pos
            auto_hedge_trade = -net_delta_after_trade
            cost_auto_hedge = self._execute_trade('underlying_pos', self.max_underlying, auto_hedge_trade, price)
        
        # Advance time and MTM
        prev_total_value = self.total_value
        self._current_index += 1
        if self._current_index >= self.expiry_index or self._current_index >= len(self.df):
            self.done = True
        
        obs_after_trade = self._get_obs()
        self.total_value = obs_after_trade['total_value']
        pnl = self.total_value - prev_total_value
        self.value_history.append(self.total_value)

        # Handle expiry
        if self._current_index >= self.expiry_index:
            # Settle options position
            final_price = self.df.iloc[self.expiry_index]['close']
            intrinsic_value = max(0, final_price - self.strike) if self.option_type == 'call' else max(0, self.strike - final_price)
            self.cash += self.option_pos * intrinsic_value
            self.option_pos = 0

        # Reward shaping
        reward = float(pnl)
        if self.reward_shaper:
            reward = self.reward_shaper(self, pnl)

        self.info = {
            'total_value': self.total_value, 'pnl': float(pnl),
            'margin_requirement': obs_after_trade['margin_requirement'],
            'underlying_pos': self.underlying_pos, 'option_pos': self.option_pos, 'cash': self.cash,
            'cost_option_trade': float(cost_o), 'cost_underlying_trade': float(cost_u),
            'cost_auto_hedge': float(cost_auto_hedge), 'auto_hedge_trade': float(auto_hedge_trade)
        }
        
        return obs_after_trade, reward, self.done, False, self.info