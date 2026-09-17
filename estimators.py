import numpy as np
import pandas as pd
import statsmodels.api as sm

def estimate_disposition_effect(trades_df, daily_portfolios, market_prices):
    """
    Calcula PGR y PLR revisando el portafolio EXACTO de cada trader
    únicamente en los días que ejecutó una venta.
    """
    if trades_df.empty:
        return {'PGR': 0, 'PLR': 0, 'diff': 0, 'ratio': 0}
        
    sales = trades_df[trades_df['type'] == 'SELL']
    if sales.empty:
        return {'PGR': 0, 'PLR': 0, 'diff': 0, 'ratio': 0}
    
    RG, RL, PG, PL = 0, 0, 0, 0
    sale_events = sales.groupby(['day', 'agent_id'])
    
    for (day, agent_id), group in sale_events:
        price_today = market_prices.iloc[day]
        
        # 1. Realized Gains & Losses (de los activos vendidos hoy)
        for _, row in group.iterrows():
            exec_p = row['execution_price']
            buy_p = row['buy_price']
            if exec_p > buy_p:
                RG += 1
            elif exec_p < buy_p:
                RL += 1
                
        # 2. Paper Gains & Losses (del PORTAFOLIO QUE LE QUEDÓ ese día)
        trader_holdings = daily_portfolios[day][agent_id]['holdings']
        for asset, data in trader_holdings.items():
            cur_p = price_today[asset]
            buy_p = data['buy_price']
            if cur_p > buy_p:
                PG += 1
            elif cur_p < buy_p:
                PL += 1
                
    PGR = RG / (RG + PG) if (RG + PG) > 0 else 0
    PLR = RL / (RL + PL) if (RL + PL) > 0 else 0
    
    return {
        'PGR': PGR, 'PLR': PLR,
        'diff': PGR - PLR,
        'ratio': PGR / PLR if PLR > 0 else np.nan
    }

def estimate_overconfidence(traders, trades_df, daily_values_gross, daily_values_net):
    """
    Corre dos regresiones OLS (Gross y Net) incluyendo controles:
    - Tamaño de portafolio inicial
    - Número final de posiciones
    - Exposición a riesgo (Volatilidad diaria del portafolio)
    """
    n_traders = len(traders)
    
    # Retornos totales
    returns_gross = (daily_values_gross[-1, :] - daily_values_gross[0, :]) / daily_values_gross[0, :]
    returns_net = (daily_values_net[-1, :] - daily_values_net[0, :]) / daily_values_net[0, :]
    
    # Cálculo de métricas por trader
    turnover_list = []
    initial_sizes = []
    n_final_positions = []
    portfolio_volatilities = []
    
    for i in range(n_traders):
        # Turnover
        trader_trades = trades_df[trades_df['agent_id'] == i] if not trades_df.empty else pd.DataFrame()
        total_volume = (trader_trades['quantity'] * trader_trades['execution_price']).sum() if not trader_trades.empty else 0
        portfolio_size = traders[i].initial_portfolio_value
        turnover = total_volume / portfolio_size
        turnover_list.append(turnover)
        
        # Controles
        initial_sizes.append(portfolio_size)
        n_final_positions.append(len(traders[i].portfolio))
        
        # Medida de Riesgo: Volatilidad de los retornos diarios del portafolio
        daily_rets = np.diff(daily_values_gross[:, i]) / daily_values_gross[:-1, i]
        portfolio_volatilities.append(np.std(daily_rets))
        
    # Dataframe de la Regresión con Controles X_i
    df_reg = pd.DataFrame({
        'return_gross': returns_gross,
        'return_net': returns_net,
        'turnover': turnover_list,
        'portfolio_size': initial_sizes,
        'num_positions': n_final_positions,
        'risk_exposure': portfolio_volatilities,
        'const': 1.0
    })
    
    X = df_reg[['const', 'turnover', 'portfolio_size', 'num_positions', 'risk_exposure']]
    
    # Regresión 1: Retornos Brutos
    model_gross = sm.OLS(df_reg['return_gross'], X).fit()
    beta_gross = model_gross.params['turnover']
    
    # Regresión 2: Retornos Netos
    model_net = sm.OLS(df_reg['return_net'], X).fit()
    beta_net = model_net.params['turnover']
    
    return beta_gross, beta_net, df_reg