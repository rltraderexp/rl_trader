"""SVI surface fitter and spline-based smoother for implied vol surfaces.

This module provides:
- SviSurface: fits SVI parameters per expiry (total variance slice) and interpolates across expiries.
- SplineSurface: fits a smooth surface using scipy's RectBivariateSpline on log-moneyness and tau.

References:
- Gatheral, "The Volatility Surface" (SVI parameterization)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import warnings
from .black_scholes import implied_vol

try:
    from scipy.optimize import least_squares
    from scipy.interpolate import RectBivariateSpline, UnivariateSpline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def _total_variance_from_iv(iv: float, tau: float) -> float:
    return (iv ** 2) * tau

def _iv_from_total_variance(w: float, tau: float) -> float:
    return np.sqrt(max(w, 0.0) / max(tau, 1e-9))

def _log_moneyness(K: float, F: float) -> float:
    return np.log(K / max(F, 1e-9))

def svi_w(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """SVI total variance function w(k)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def _svi_residuals(params, ks, ws):
    a, b, rho, m, sigma = params
    # Penalty for invalid parameters
    if b < 0 or sigma <= 0 or abs(rho) >= 1.0:
        return 1e6 * np.ones_like(ws)
    return svi_w(ks, a, b, rho, m, sigma) - ws

def fit_svi_slice(ks: np.ndarray, ivs: np.ndarray, tau: float):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for SVI fitting")
    ws = _total_variance_from_iv(ivs, tau)
    
    # Robust initial guess
    a0 = np.median(ws) * 0.8
    b0 = max(0.01, np.std(ws) / (np.std(ks) + 1e-6))
    rho0, m0, sigma0 = -0.5, np.median(ks), 0.1
    x0 = np.array([a0, b0, rho0, m0, sigma0])

    res = least_squares(_svi_residuals, x0, args=(ks, ws), method='trf', ftol=1e-8, xtol=1e-8, max_nfev=2000)
    p = res.x
    # Ensure params are valid
    p[1] = max(p[1], 1e-8)  # b >= 0
    p[2] = np.clip(p[2], -0.999, 0.999) # |rho| < 1
    p[4] = max(p[4], 1e-8) # sigma > 0
    return tuple(p)

class SviSurface:
    def __init__(self, taus: List[float], params_by_tau: Dict[float, tuple], forward_by_tau: Dict[float, float]):
        self.taus = np.asarray(sorted(taus))
        self.params_by_tau = params_by_tau
        self.forward_by_tau = forward_by_tau
        self._param_interpolators = {}
        self._fit_param_interpolators()

    def _fit_param_interpolators(self):
        if not SCIPY_AVAILABLE or len(self.taus) < 2:
            return
        params = np.array([self.params_by_tau[t] for t in self.taus])
        for i, name in enumerate(['a', 'b', 'rho', 'm', 'sigma']):
            self._param_interpolators[name] = UnivariateSpline(self.taus, params[:, i], s=0, k=1)
        forwards = np.array([self.forward_by_tau[t] for t in self.taus])
        self._param_interpolators['forward'] = UnivariateSpline(self.taus, forwards, s=0, k=1)

    @classmethod
    def from_quotes(cls, quotes: List[Dict[str, Any]]):
        from ..data.market_data import estimate_forward_from_chain
        
        df = pd.DataFrame(quotes)
        by_tau = {tau: group for tau, group in df.groupby('tau')}
        
        params, forwards = {}, {}
        for tau, group in by_tau.items():
            S = group['spot'].iloc[0]
            F, _ = estimate_forward_from_chain(group, assume_r=group['r'].iloc[0])
            
            ivs, ks = [], []
            for _, row in group.iterrows():
                try:
                    iv = implied_vol(row['spot'], row['strike'], row['r'], row['price'], tau, row['type'])
                    if np.isfinite(iv) and iv > 1e-4:
                        ivs.append(iv)
                        ks.append(_log_moneyness(row['strike'], F))
                except Exception:
                    continue
            
            if len(ivs) > 4:
                params[tau] = fit_svi_slice(np.array(ks), np.array(ivs), tau)
                forwards[tau] = F

        return cls(list(params.keys()), params, forwards)

    def interp_params(self, tau: float) -> Tuple[tuple, float]:
        if tau in self.params_by_tau:
            return self.params_by_tau[tau], self.forward_by_tau[tau]
        if not self._param_interpolators or tau < self.taus[0] or tau > self.taus[-1]:
            # Extrapolate flat
            idx = np.argmin(np.abs(self.taus - tau))
            t = self.taus[idx]
            return self.params_by_tau[t], self.forward_by_tau[t]

        p = tuple(self._param_interpolators[name](tau) for name in ['a','b','rho','m','sigma'])
        F = self._param_interpolators['forward'](tau)
        return p, float(F)

    def interp_iv(self, K: float, tau: float) -> float:
        params, F = self.interp_params(tau)
        k = _log_moneyness(K, F)
        w = svi_w(k, *params)
        return _iv_from_total_variance(w, tau)

class SplineSurface:
    def __init__(self, ks_grid, taus_grid, iv_grid, forward):
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy required for SplineSurface")
        self.ks_grid = np.asarray(ks_grid)
        self.taus_grid = np.asarray(taus_grid)
        self.iv_grid = np.asarray(iv_grid)
        self.forward = float(forward)
        self._spline = RectBivariateSpline(self.ks_grid, self.taus_grid, self.iv_grid, kx=3, ky=3)

    def interp_iv(self, K: float, tau: float) -> float:
        k = _log_moneyness(K, self.forward)
        iv = self._spline(k, tau, grid=False)
        return float(iv)