"""Implied volatility surface builder and interpolator.
Builds a simple grid of implied volatilities from option quotes (using Newton-Raphson implied_vol),
fills missing grid points with nearest-neighbor, and provides bilinear interpolation.
This is a lightweight utility; for production consider using more advanced surface fitting (SABR, SVI).
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from .black_scholes import implied_vol

class IVSurface:
    def __init__(self, strikes: List[float], taus: List[float], iv_grid: np.ndarray):
        """
        strikes: sorted list of strike prices (length M)
        taus: sorted list of times-to-expiry in years (length N)
        iv_grid: M x N array of implied vol values
        """
        self.strikes = np.asarray(strikes, dtype=float)
        self.taus = np.asarray(taus, dtype=float)
        self.iv_grid = np.asarray(iv_grid, dtype=float)
        assert self.iv_grid.shape == (len(self.strikes), len(self.taus))

    @classmethod
    def from_quotes(cls, quotes: List[Dict[str, Any]], strikes: List[float]=None, taus: List[float]=None):
        """
        quotes: list of {'strike':K, 'tau':tau_years, 'price':market_price, 'type':'call'/'put', 'spot':S, 'r':r}
        strikes, taus: optional grids to evaluate; if None grid is deduced from quotes
        """
        Ks = sorted(list(set([q['strike'] for q in quotes]))) if strikes is None else strikes
        Ts = sorted(list(set([q['tau'] for q in quotes]))) if taus is None else taus
        grid = np.full((len(Ks), len(Ts)), np.nan, dtype=float)

        # Map quotes to the grid
        for q in quotes:
            try:
                # Find nearest grid indices
                i = np.argmin(np.abs(np.asarray(Ks) - q['strike']))
                j = np.argmin(np.abs(np.asarray(Ts) - q['tau']))
                if np.abs(Ks[i] - q['strike']) < 1e-5 and np.abs(Ts[j] - q['tau']) < 1e-8:
                    iv = implied_vol(q['spot'], q['strike'], q.get('r', 0.0), q['price'], q['tau'], option_type=q.get('type','call'))
                    grid[i, j] = iv
            except (ValueError, RuntimeError):
                continue # Skip if implied_vol fails

        # fill NaNs by nearest neighbor
        filled_grid = cls._fill_nans_nearest(grid)
        return cls(Ks, Ts, filled_grid)

    @staticmethod
    def _fill_nans_nearest(grid: np.ndarray) -> np.ndarray:
        """
        Fills NaNs using nearest-neighbor interpolation.
        Note: This can be computationally intensive for large grids. A k-d tree (e.g., scipy.spatial.cKDTree)
        would be a more efficient approach for large-scale problems.
        """
        M, N = grid.shape
        coords = np.argwhere(~np.isnan(grid))
        if coords.shape[0] == 0:
            return np.full_like(grid, 0.2) # Default to 20% vol if no data

        nan_coords = np.argwhere(np.isnan(grid))
        
        # For each NaN, find the closest coordinate with a value
        for i_nan, j_nan in nan_coords:
            distances = np.sqrt((coords[:, 0] - i_nan)**2 + (coords[:, 1] - j_nan)**2)
            nearest_idx = np.argmin(distances)
            nearest_coord = coords[nearest_idx]
            grid[i_nan, j_nan] = grid[nearest_coord[0], nearest_coord[1]]
            
        return grid

    def interp_iv(self, K: float, tau: float) -> float:
        """Bilinear interpolation on the strike-tau grid. Extrapolates using nearest edges if outside grid."""
        Ks, Ts, G = self.strikes, self.taus, self.iv_grid
        
        K, tau = float(K), float(tau)

        # Find indices for strike and tau, handling extrapolation by clamping
        i1 = np.searchsorted(Ks, K, side='right')
        j1 = np.searchsorted(Ts, tau, side='right')
        
        i0 = max(0, i1 - 1)
        j0 = max(0, j1 - 1)
        i1 = min(len(Ks) - 1, i1)
        j1 = min(len(Ts) - 1, j1)

        # Calculate weights for interpolation
        dK = (Ks[i1] - Ks[i0])
        wK = (K - Ks[i0]) / dK if dK > 1e-9 else 0.0

        dT = (Ts[j1] - Ts[j0])
        wT = (tau - Ts[j0]) / dT if dT > 1e-9 else 0.0
        
        # Clamp weights to [0, 1] for extrapolation
        wK = np.clip(wK, 0.0, 1.0)
        wT = np.clip(wT, 0.0, 1.0)
        
        # Bilinear interpolation formula
        v00, v10, v01, v11 = G[i0, j0], G[i1, j0], G[i0, j1], G[i1, j1]
        iv = v00 * (1 - wK) * (1 - wT) + \
             v10 * wK * (1 - wT) + \
             v01 * (1 - wK) * wT + \
             v11 * wK * wT
             
        return float(iv)