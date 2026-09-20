import numpy as np
import pandas as pd
from agents import Agent
from config import N_TRADERS, SPREAD_BPS, COMMISSION_USD

def run_scenario(scenario_cfg, market_prices, n_traders=N_TRADERS):
    """Ejecuta un escenario de simulación completo con N agentes a lo largo del tiempo.

    Args:
        scenario_cfg (dict): Configuración del escenario ('alpha_disp', 'alpha_over', 'confound').
        market_prices (pd.DataFrame): Trayectoria de precios de mercado.
        n_traders (int, optional): Número de traders a generar. Por defecto N_TRADERS.

    Returns:
        tuple:
            - list[Agent]: Lista con las instancias de los agentes.
            - pd.DataFrame: Registro histórico de todas las operaciones ejecutadas.
            - np.ndarray: Matriz de valores brutos diarios por agente.
            - np.ndarray: Matriz de valores netos diarios por agente.
            - dict: Registro de fotogramas diarios del estado del portafolio de cada agente.
    """
    alpha_disp_val = scenario_cfg['alpha_disp']
    alpha_over_val = scenario_cfg['alpha_over']
    confound = scenario_cfg['confound']
    
    n_days, n_assets = market_prices.shape
    traders = []
    
    # --- 1. GENERACIÓN E INICIALIZACIÓN DE TRADERS ---
    for i in range(n_traders):
        a_disp = np.random.uniform(0, 1) if alpha_disp_val == "rand" else float(alpha_disp_val)
        a_over = np.random.uniform(0, 1) if alpha_over_val == "rand" else float(alpha_over_val)
        initial_cash = np.random.uniform(10000, 500000)
        
        n_pos = int(np.random.uniform(5, 31))
        traders.append(Agent(i, a_disp, a_over, initial_cash, n_assets, n_initial_positions=n_pos))
        
    all_trades = []
    daily_values_gross = np.zeros((n_days, n_traders))
    daily_values_net = np.zeros((n_days, n_traders))
    daily_portfolios = {} 

    # --- 2. BUCLE PRINCIPAL DE DÍAS Y VALORACIÓN DE PORTAFOLIOS ---
    for t in range(n_days):
        current_prices = market_prices.iloc[t]
        daily_portfolios[t] = {}
        
        for idx, trader in enumerate(traders):
            trades = trader.decide_trades(t, current_prices, spread_bps=SPREAD_BPS, commission_usd=COMMISSION_USD, confound=confound)
            all_trades.extend(trades)
            
            # Registro de estado diario del portafolio
            daily_portfolios[t][trader.id] = {
                'cash_net': trader.cash_net,
                'cash_gross': trader.cash_gross,
                'holdings': {k: v.copy() for k, v in trader.portfolio.items()}
            }
            
            # Valor Bruto: Sin fricciones bursátiles
            pos_gross = sum(
                data['quantity'] * (current_prices.iloc[asset] if isinstance(asset, (int, np.integer)) else current_prices.loc[asset]) 
                for asset, data in trader.portfolio.items()
            )
            daily_values_gross[t, idx] = trader.cash_gross + pos_gross
            
            # Valor Neto: Con spreads Bid-Ask aplicados
            half_spread = (SPREAD_BPS / 10000.0) / 2.0
            pos_net = sum(
                data['quantity'] * ((current_prices.iloc[asset] if isinstance(asset, (int, np.integer)) else current_prices.loc[asset]) * (1.0 - half_spread)) 
                for asset, data in trader.portfolio.items()
            )
            daily_values_net[t, idx] = trader.cash_net + pos_net
            
    trades_df = pd.DataFrame(all_trades)
    
    return traders, trades_df, daily_values_gross, daily_values_net, daily_portfolios