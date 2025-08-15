"""Reward shaping utilities for environments."""
from typing import Optional
import numpy as np

def vol_penalty_shaper(env, pnl: float, window: int = 30, risk_aversion: float = 0.1) -> float:
    """Computes reward = pnl - risk_aversion * volatility * portfolio_value"""
    hist = getattr(env, 'value_history', None)
    if not hist or len(hist) < 2:
        return float(pnl)
        
    current_window = min(window, len(hist) - 1)
    if current_window < 1:
        return float(pnl)
        
    vals = np.array(hist[-current_window-1:], dtype=float)
    returns = np.diff(vals) / (vals[:-1] + 1e-12)
    volatility = np.std(returns)
    
    current_value = float(env.total_value)
    penalty = float(risk_aversion) * volatility * abs(current_value)
    
    return float(pnl) - penalty

class DifferentialSharpeShaper:
    """
    Calculates a reward based on the incremental change to the Sharpe ratio.
    Reference: Moody & Saffell (2001), "Learning to Trade via Direct Reinforcement".
    """
    def __init__(self, learning_rate: float = 0.01, target_sharpe: float = 0.0):
        self.learning_rate = float(learning_rate)
        self.target_sharpe = float(target_sharpe)
        self.A = 0.0
        self.B = 1.0
        self.sharpe = 0.0

    def __call__(self, env, pnl: float) -> float:
        # For simplicity, we use PnL as the return for the update
        ret = pnl / (env.total_value - pnl + 1e-9) if (env.total_value - pnl) != 0 else 0
        
        # Update exponential moving averages of the first and second moments of returns
        self.A = (1 - self.learning_rate) * self.A + self.learning_rate * ret
        self.B = (1 - self.learning_rate) * self.B + self.learning_rate * (ret ** 2)
        
        # Avoid division by zero for standard deviation
        std_dev = np.sqrt(self.B - self.A**2)
        if std_dev < 1e-9:
            return 0.0
        
        # Calculate the differential Sharpe reward
        prev_sharpe = self.sharpe
        self.sharpe = self.A / std_dev
        
        reward = (ret - prev_sharpe) / std_dev
        return float(reward)