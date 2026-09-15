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
    daily_values_gross = np.zeros((n_days, n_traders))
    
    for t in range(n_days):
        current_prices = market_prices.iloc[t]
        for idx, trader in enumerate(traders):
            trades = trader.decide_trades(t, current_prices, spread_bps=SPREAD_BPS, commission_usd=COMMISSION_USD, confound=confound)
            all_trades.extend(trades)
            daily_values_gross[t, idx] = trader.evaluate_portfolio(current_prices)
            
    trades_df = pd.DataFrame(all_trades)
    return traders, trades_df, daily_values_gross