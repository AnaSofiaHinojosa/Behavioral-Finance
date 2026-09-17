import numpy as np
import pandas as pd
from estimators import estimate_disposition_effect, estimate_overconfidence
from config import N_BOOTSTRAP

def run_cluster_bootstrap(traders, trades_df, daily_portfolios, market_prices, daily_values_gross, daily_values_net, n_boot=N_BOOTSTRAP):
    n_traders = len(traders)
    disp_diffs, over_betas_gross = [], []
    agent_ids = np.arange(n_traders)
    
    for _ in range(n_boot):
        boot_agents = np.random.choice(agent_ids, size=n_traders, replace=True)
        boot_trades_list = []
        
        for new_id, orig_id in enumerate(boot_agents):
            if not trades_df.empty:
                t_df = trades_df[trades_df['agent_id'] == orig_id].copy()
                t_df['agent_id'] = new_id
                boot_trades_list.append(t_df)
                
        boot_trades = pd.concat(boot_trades_list) if boot_trades_list else pd.DataFrame()
        boot_values_gross = daily_values_gross[:, boot_agents]
        boot_values_net = daily_values_net[:, boot_agents]
        
        disp_res = estimate_disposition_effect(boot_trades, daily_portfolios, market_prices)
        beta_gross, beta_net, _ = estimate_overconfidence([traders[i] for i in boot_agents], boot_trades, boot_values_gross, boot_values_net)
        
        disp_diffs.append(disp_res['diff'])
        over_betas_gross.append(beta_gross)
        
    return np.std(disp_diffs), np.std(over_betas_gross)