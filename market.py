import numpy as np
import pandas as pd
from config import N_ASSETS, N_DAYS, S0, MU, SIGMA, DT

def generate_market_data(n_assets=N_ASSETS, n_days=N_DAYS, s0=S0, mu=MU, sigma=SIGMA, dt=DT, seed=None):
    """Genera precios de activos mediante Geometric Brownian Motion (GBM)."""
    if seed is not None:
        np.random.seed(seed)
        
    price_paths = np.zeros((n_days, n_assets))
    price_paths[0] = s0
    
    for t in range(1, n_days):
        drift = (mu - 0.5 * sigma**2) * dt
        shock = sigma * np.sqrt(dt) * np.random.normal(0, 1, n_assets)
        price_paths[t] = price_paths[t-1] * np.exp(drift + shock)
        
    columns = [f"Asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(price_paths, columns=columns)