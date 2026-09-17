import numpy as np
import pandas as pd
from agents import Agent
from config import N_TRADERS, SPREAD_BPS, COMMISSION_USD

def run_scenario(scenario_cfg, market_prices, n_traders=N_TRADERS):
    alpha_disp_val = scenario_cfg['alpha_disp']
    alpha_over_val = scenario_cfg['alpha_over']
    confound = scenario_cfg['confound']
    
    n_days, n_assets = market_prices.shape
    traders = []
    
    for i in range(n_traders):
        a_disp = np.random.uniform(0, 1) if alpha_disp_val == "rand" else alpha_disp_val
        a_over = np.random.uniform(0, 1) if alpha_over_val == "rand" else alpha_over_val
        initial_cash = np.random.uniform(10000, 500000)
        traders.append(Agent(i, a_disp, a_over, initial_cash, n_assets))
        
    all_trades = []
    
    # Tracking diario de valores brutos y netos
    daily_values_gross = np.zeros((n_days, n_traders))
    daily_values_net = np.zeros((n_days, n_traders))
    
    # Registro del estado del portafolio por trader y por día
    daily_portfolios = {} 

    for t in range(n_days):
        current_prices = market_prices.iloc[t]
        daily_portfolios[t] = {}
        
        for idx, trader in enumerate(traders):
            trades = trader.decide_trades(t, current_prices, spread_bps=SPREAD_BPS, commission_usd=COMMISSION_USD, confound=confound)
            all_trades.extend(trades)
            
            # Guardar foto exacta del portafolio del trader al final del día
            daily_portfolios[t][trader.id] = {
                'cash': trader.cash,
                'holdings': {k: v.copy() for k, v in trader.portfolio.items()}
            }
            
            # Valor Bruto (al precio puro de mercado sin fricciones)
            pos_gross = sum(data['quantity'] * current_prices[asset] for asset, data in trader.portfolio.items())
            daily_values_gross[t, idx] = trader.cash + pos_gross
            
            # Valor Neto (valuado a precio Bid por si tuviera que liquidar hoy)
            pos_net = sum(data['quantity'] * (current_prices[asset] * (1.0 - (SPREAD_BPS/10000.0)/2)) for asset, data in trader.portfolio.items())
            daily_values_net[t, idx] = trader.cash + pos_net
            
    trades_df = pd.DataFrame(all_trades)
    
    return traders, trades_df, daily_values_gross, daily_values_net, daily_portfolios