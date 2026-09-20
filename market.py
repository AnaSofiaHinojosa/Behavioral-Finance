import numpy as np
import pandas as pd
from config import N_ASSETS, N_DAYS, S0, MU, SIGMA, DT

def generate_market_data(n_assets=N_ASSETS, n_days=N_DAYS, s0=S0, mu=MU, sigma=SIGMA, dt=DT, seed=None):
    """Genera precios sintéticos de activos mediante Movimiento Browniano Geométrico (GBM).

    Args:
        n_assets (int, optional): Número total de activos a simular. Por defecto N_ASSETS.
        n_days (int, optional): Número de días de trading. Por defecto N_DAYS.
        s0 (float, optional): Precio inicial de todos los activos en t=0. Por defecto S0.
        mu (float, optional): Drift o retorno esperado anualizado. Por defecto MU.
        sigma (float, optional): Volatilidad anualizada. Por defecto SIGMA.
        dt (float, optional): Incremento de tiempo por paso. Por defecto DT.
        seed (int, optional): Semilla para el generador de números aleatorios. Por defecto None.

    Returns:
        pd.DataFrame: DataFrame de forma (n_days, n_assets) con las trayectorias de precios.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # --- 1. INICIALIZACIÓN DE MATRIZ DE PRECIOS ---
    price_paths = np.zeros((n_days, n_assets))
    price_paths[0] = s0
    
    # --- 2. SIMULACIÓN DE TRAYECTORIAS ESTOCÁSTICAS (GBM) ---
    for t in range(1, n_days):
        drift = (mu - 0.5 * sigma**2) * dt
        shock = sigma * np.sqrt(dt) * np.random.normal(0, 1, n_assets)
        price_paths[t] = price_paths[t-1] * np.exp(drift + shock)
        
    columns = [f"Asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(price_paths, columns=columns)