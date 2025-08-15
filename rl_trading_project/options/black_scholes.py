"""Black-Scholes pricing and Greeks for European options."""
import math
import numpy as np
from scipy.stats import norm

SQRT_2PI = math.sqrt(2 * math.pi)

def bs_price(S, K, r, sigma, tau, option_type='call', q=0.0):
    if tau <= 1e-9 or sigma <= 1e-9:
        if option_type == 'call':
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    if option_type == 'call':
        price = S * math.exp(-q * tau) * norm.cdf(d1) - K * math.exp(-r * tau) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * tau) * norm.cdf(-d2) - S * math.exp(-q * tau) * norm.cdf(-d1)
    return float(price)

def bs_vega(S, K, r, sigma, tau, q=0.0):
    if tau <= 1e-9 or sigma <= 1e-9:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    pdf_d1 = math.exp(-0.5 * d1 * d1) / SQRT_2PI
    return float(S * math.exp(-q * tau) * pdf_d1 * math.sqrt(tau))

def bs_delta(S, K, r, sigma, tau, option_type='call', q=0.0):
    """Returns Black-Scholes delta (European)."""
    if tau <= 1e-9 or sigma <= 1e-9:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    if option_type == 'call':
        return float(math.exp(-q * tau) * norm.cdf(d1))
    else:
        return float(math.exp(-q * tau) * (norm.cdf(d1) - 1.0))

def bs_gamma(S, K, r, sigma, tau, q=0.0):
    if tau <= 1e-9 or sigma <= 1e-9:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    pdf_d1 = math.exp(-0.5 * d1 * d1) / SQRT_2PI
    return float(pdf_d1 * math.exp(-q * tau) / (S * sigma * math.sqrt(tau)))

def bs_theta(S, K, r, sigma, tau, option_type='call', q=0.0):
    """Approximate Theta per year (negative is decay)."""
    if tau <= 1e-9 or sigma <= 1e-9:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    pdf_d1 = math.exp(-0.5 * d1 * d1) / SQRT_2PI
    
    first_term = - (S * pdf_d1 * sigma * math.exp(-q * tau)) / (2 * math.sqrt(tau))
    if option_type == 'call':
        second_term = -r * K * math.exp(-r * tau) * norm.cdf(d2)
        third_term = q * S * math.exp(-q * tau) * norm.cdf(d1)
        theta = first_term + second_term + third_term
    else: # put
        second_term = r * K * math.exp(-r * tau) * norm.cdf(-d2)
        third_term = -q * S * math.exp(-q * tau) * norm.cdf(-d1)
        theta = first_term + second_term + third_term
    return float(theta)

def implied_vol(S, K, r, price, tau, option_type='call', q=0.0, tol=1e-6, maxiter=100):
    # Use Newton-Raphson on volatility
    sigma = 0.2 # Initial guess
    for i in range(maxiter):
        price_est = bs_price(S, K, r, sigma, tau, option_type=option_type, q=q)
        diff = price_est - price
        if abs(diff) < tol:
            return float(max(sigma, 1e-12))
        vega = bs_vega(S, K, r, sigma, tau, q=q)
        if vega < 1e-9:
            break # Vega is too small, cannot update
        sigma = sigma - diff / vega
        sigma = max(sigma, 1e-12) # Ensure sigma stays positive
    return float(sigma)