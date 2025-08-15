"""
Simple Order Book Execution Simulator

This module provides a lightweight model to simulate execution using a
liquidity profile rather than using full LOB snapshots. It supports:
- Market orders with market-impact slippage (linear & square-root models)
- Limit orders that may be partially filled depending on opposing liquidity
- Simple liquidity curve parameterization: depth (notional at BID/ASK),
  impact_coefficient controlling price impact per unit traded.

The goal is to provide higher-fidelity fills than a fixed slippage fraction,
suitable for agent training where fill/partial-fill behavior matters.
"""

# existing imports kept...
from typing import Tuple, Dict, Any, Union
import math
import numpy as np

def market_impact_price(price: float, trade_notional: float, liquidity_depth: float, model: str = "sqrt") -> float:
    # unchanged from original, kept for scalar fallback
    if trade_notional == 0 or liquidity_depth <= 0:
        return price
    sign = 1.0 if trade_notional > 0 else -1.0
    q = abs(trade_notional)
    if model == "linear":
        impact_factor = (q / liquidity_depth)
    else:
        impact_factor = math.sqrt(q / liquidity_depth)
    impact_factor = min(impact_factor, 0.5)
    exec_price = price * (1.0 + sign * impact_factor)
    return exec_price

def simulate_market_order(price: float, size: float, liquidity_depth: float, commission: float = 0.0, model: str="sqrt") -> Dict[str, Any]:
    # unchanged scalar implementation (kept for compatibility)
    if size == 0:
        return {'filled_size': 0, 'avg_price': price, 'commission': 0, 'slippage_notional': 0}
    notional = abs(size) * price
    exec_price = market_impact_price(price, size * price, liquidity_depth, model=model)
    commission_cost = abs(size * exec_price) * commission
    slippage_notional = abs(size * (exec_price - price))
    return {
        'filled_size': size,
        'avg_price': exec_price,
        'commission': commission_cost,
        'slippage_notional': slippage_notional
    }

# -------------------------
# New: batched simulation for vectorized envs (numpy arrays)
# -------------------------
def simulate_market_order_batch(prices: Union[np.ndarray, float],
                                sizes: Union[np.ndarray, float],
                                liquidity_depths: Union[np.ndarray, float],
                                commissions: Union[np.ndarray, float] = 0.0,
                                model: str = "sqrt") -> Dict[str, np.ndarray]:
    """
    Vectorized simulate_market_order for arrays of orders.
    Inputs can be scalars or 1D numpy arrays; outputs are numpy arrays of same shape.
    Returns dict with keys:
      - 'filled_size' (array), 'avg_price' (array), 'commission' (array), 'slippage_notional' (array)
    """
    prices = np.asarray(prices, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    liquidity_depths = np.asarray(liquidity_depths, dtype=float)
    commissions = np.asarray(commissions, dtype=float)

    # broadcast to same shape
    shapes = np.broadcast_shapes(prices.shape, sizes.shape, liquidity_depths.shape, commissions.shape)
    prices = np.broadcast_to(prices, shapes).astype(float)
    sizes = np.broadcast_to(sizes, shapes).astype(float)
    liquidity_depths = np.broadcast_to(liquidity_depths, shapes).astype(float)
    commissions = np.broadcast_to(commissions, shapes).astype(float)

    filled = np.zeros_like(sizes)
    avg_price = np.zeros_like(prices)
    commission_cost = np.zeros_like(prices)
    slippage_notional = np.zeros_like(prices)

    # mask for nonzero orders
    nz = sizes != 0
    if nz.any():
        sign = np.sign(sizes[nz])
        q = np.abs(sizes[nz]) * prices[nz]  # notional
        # compute impact factor vectorized
        if model == "linear":
            impact_factor = (q / (liquidity_depths[nz] + 1e-12))
        else:
            impact_factor = np.sqrt(np.maximum(q / (liquidity_depths[nz] + 1e-12), 0.0))
        impact_factor = np.minimum(impact_factor, 0.5)
        exec_price_nz = prices[nz] * (1.0 + sign * impact_factor)
        filled[nz] = sizes[nz]
        avg_price[nz] = exec_price_nz
        commission_cost[nz] = np.abs(filled[nz] * avg_price[nz]) * commissions[nz]
        slippage_notional[nz] = np.abs(filled[nz] * (avg_price[nz] - prices[nz]))

    return {
        'filled_size': filled,
        'avg_price': avg_price,
        'commission': commission_cost,
        'slippage_notional': slippage_notional
    }