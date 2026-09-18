import numpy as np
import pandas as pd
from estimators import estimate_disposition_effect, estimate_overconfidence
from config import N_BOOTSTRAP

def run_cluster_bootstrap(traders, trades_df, daily_portfolios, market_prices, daily_values_gross, daily_values_net, n_boot=N_BOOTSTRAP):
    n_traders = len(traders)
    disp_diffs, over_betas_gross = [], []
    agent_ids = np.arange(n_traders)
    n_days = len(daily_portfolios)
    
    for _ in range(n_boot):
        boot_agents = np.random.choice(agent_ids, size=n_traders, replace=True)
        boot_trades_list = []
        
        # Mapeo y reindexación de portafolios
        boot_daily_portfolios = {day: {} for day in range(n_days)}
        
        for new_id, orig_id in enumerate(boot_agents):
            # 1. Reindexar Trades
            if not trades_df.empty:
                t_df = trades_df[trades_df['agent_id'] == orig_id].copy()
                t_df['agent_id'] = new_id
                boot_trades_list.append(t_df)
                
            # 2. Reindexar Fotos Diarias del Portafolio del Agente
            for day in range(n_days):
                boot_daily_portfolios[day][new_id] = daily_portfolios[day][orig_id]
                
        boot_trades = pd.concat(boot_trades_list) if boot_trades_list else pd.DataFrame()
        boot_values_gross = daily_values_gross[:, boot_agents]
        boot_values_net = daily_values_net[:, boot_agents]
        
        disp_res = estimate_disposition_effect(boot_trades, boot_daily_portfolios, market_prices)
        beta_gross, _, _ = estimate_overconfidence([traders[i] for i in boot_agents], boot_trades, boot_values_gross, boot_values_net)
        
        disp_diffs.append(disp_res['diff'])
        over_betas_gross.append(beta_gross)
        
    return np.std(disp_diffs), np.std(over_betas_gross)