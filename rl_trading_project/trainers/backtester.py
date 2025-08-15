"""Backtester and evaluation harness for options strategies using OptionsEnv."""
from typing import Callable, Dict, Any, List, Tuple
import numpy as np
import math
import time
import csv
from collections import defaultdict

class Backtester:
    def __init__(self, env_factory: Callable[[], Any], start_index: int = 0, end_index: int = None):
        """
        env_factory: callable returning a fresh environment instance.
        start_index/end_index: optional indices to limit the backtest horizon.
        """
        self.env_factory = env_factory
        self.start_index = start_index
        self.end_index = end_index

    def run(self, policy_fn: Callable[[Dict[str,Any], int], List[float]], max_steps: int = 1000):
        """
        Run a backtest where policy_fn(obs, t) -> action.
        Returns a results dict with timeseries metrics and summary statistics.
        """
        env = self.env_factory()
        obs, _ = env.reset(start_index=self.start_index)
        t = 0
        history = []
        start_time = time.time()
        
        # Determine max steps from environment or argument
        if self.end_index:
            max_steps = min(max_steps, self.end_index - self.start_index)

        done = False
        while t < max_steps and not done:
            action = policy_fn(obs, t)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            row = {
                't': t,
                'timestamp': obs.get('timestamp'),
                'reward': float(reward or 0.0),
                **{k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in info.items()}
            }
            history.append(row)
            t += 1
        
        duration = time.time() - start_time
        results = self._summarize(history, duration)
        results['history'] = history
        return results

    def _summarize(self, history: List[Dict[str,Any]], duration_s: float) -> Dict[str,Any]:
        if not history:
            return {}
        
        tv = [h['total_value'] for h in history]
        pnl = [h['pnl'] for h in history]
        
        total_return = (tv[-1] / tv[0] - 1.0) if tv[0] != 0 else 0.0
        
        # Annualized Sharpe Ratio (assuming daily steps for simplicity)
        daily_returns = np.diff(tv) / tv[:-1]
        sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-9)) * np.sqrt(252) if len(daily_returns) > 1 else 0.0
        
        # Max Drawdown
        peak = np.maximum.accumulate(tv)
        drawdown = (peak - tv) / (peak + 1e-9)
        max_drawdown = np.max(drawdown)

        return {
            'n_steps': len(history),
            'duration_s': float(duration_s),
            'start_value': float(tv[0]),
            'end_value': float(tv[-1]),
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'avg_pnl': np.mean(pnl),
            'pnl_std': np.std(pnl),
        }

def save_history_csv(history: list, path: str):
    if not history: return
    keys = sorted(list(history[0].keys()))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)

def compare_strategies(strategy_results: dict) -> dict:
    summary = {}
    for name, res in strategy_results.items():
        summary[name] = {k: v for k, v in res.items() if k != 'history'}
    return summary