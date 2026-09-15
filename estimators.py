import numpy as np
import pandas as pd
import statsmodels.api as sm

def estimate_disposition_effect(trades_df, market_prices):
    if trades_df.empty:
        return {'PGR': 0, 'PLR': 0, 'diff': 0, 'ratio': 0}
        
    sales = trades_df[trades_df['type'] == 'SELL']
    if sales.empty:
        return {'PGR': 0, 'PLR': 0, 'diff': 0, 'ratio': 0}
    
    RG, RL, PG, PL = 0, 0, 0, 0
    sale_events = sales.groupby(['day', 'agent_id'])
    
    for (day, agent_id), group in sale_events:
        price_today = market_prices.iloc[day]
        for _, row in group.iterrows():
            exec_p = row['execution_price']
            buy_p = row['buy_price']
            if exec_p > buy_p:
                RG += 1
            elif exec_p < buy_p:
                RL += 1
                
        for asset in market_prices.columns:
            cur_p = price_today[asset]
            if cur_p > buy_p:
                PG += 0.1
            else:
                PL += 0.1
                
    PGR = RG / (RG + PG) if (RG + PG) > 0 else 0
    PLR = RL / (RL + PL) if (RL + PL) > 0 else 0
    
    return {
        'PGR': PGR, 'PLR': PLR,
        'diff': PGR - PLR,
        'ratio': PGR / PLR if PLR > 0 else np.nan
    }

def estimate_overconfidence(traders, trades_df, daily_values_gross):
    n_traders = len(traders)
    returns_gross = (daily_values_gross[-1, :] - daily_values_gross[0, :]) / daily_values_gross[0, :]
    
    turnover_list = []
    for i in range(n_traders):
        trader_trades = trades_df[trades_df['agent_id'] == i] if not trades_df.empty else pd.DataFrame()
        total_volume = (trader_trades['quantity'] * trader_trades['execution_price']).sum() if not trader_trades.empty else 0
        portfolio_size = traders[i].initial_portfolio_value
        turnover = total_volume / portfolio_size
        turnover_list.append(turnover)
        
    df_reg = pd.DataFrame({
        'return_gross': returns_gross,
        'turnover': turnover_list,
        'const': 1.0
    })
    
    model_gross = sm.OLS(df_reg['return_gross'], df_reg[['const', 'turnover']]).fit()
    return model_gross.params['turnover'], df_reg