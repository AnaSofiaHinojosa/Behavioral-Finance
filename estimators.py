import numpy as np
import pandas as pd
import statsmodels.api as sm

def estimate_disposition_effect(trades_df, daily_portfolios, market_prices):
    """Estima el Disposition Effect mediante las métricas PGR y PLR (Odean, 1998).

    Args:
        trades_df (pd.DataFrame): Registro histórico de transacciones.
        daily_portfolios (dict): Registro histórico del estado diario de los portafolios.
        market_prices (pd.DataFrame): Precios históricos del mercado.

    Returns:
        dict: Diccionario con las métricas PGR, PLR, la diferencia (PGR-PLR) y la razón (PGR/PLR).
    """
    if trades_df.empty:
        return {'PGR': 0.0, 'PLR': 0.0, 'diff': 0.0, 'ratio': np.nan}
        
    sales = trades_df[trades_df['type'] == 'SELL']
    if sales.empty:
        return {'PGR': 0.0, 'PLR': 0.0, 'diff': 0.0, 'ratio': np.nan}
    
    RG, RL, PG, PL = 0, 0, 0, 0
    sale_events = sales.groupby(['day', 'agent_id'])
    
    # --- 1. CONTEO DE GANANCIAS/PÉRDIDAS REALIZADAS Y NO REALIZADAS ---
    for (day, agent_id), group in sale_events:
        price_today = market_prices.iloc[day]
        
        # Ganancias y Pérdidas Realizadas
        for _, row in group.iterrows():
            exec_p = row['execution_price']
            buy_p = row['buy_price']
            if exec_p > buy_p:
                RG += 1
            elif exec_p < buy_p:
                RL += 1
                
        # Ganancias y Pérdidas Papel (No Realizadas)
        trader_holdings = daily_portfolios[day][agent_id]['holdings']
        for asset, data in trader_holdings.items():
            cur_p = price_today.iloc[asset] if isinstance(asset, (int, np.integer)) else price_today.loc[asset]
            buy_p = data['buy_price']
            if cur_p > buy_p:
                PG += 1
            elif cur_p < buy_p:
                PL += 1
                
    # --- 2. CÁLCULO DE PROPORCIONES Y DIFERENCIAL ---
    PGR = RG / (RG + PG) if (RG + PG) > 0 else 0.0
    PLR = RL / (RL + PL) if (RL + PL) > 0 else 0.0
    
    return {
        'PGR': PGR,
        'PLR': PLR,
        'diff': PGR - PLR,
        'ratio': PGR / PLR if PLR > 0 else np.nan
    }

def estimate_overconfidence(traders, trades_df, daily_values_gross, daily_values_net):
    """Estima el impacto del Overconfidence evaluando el retorno vs. turnover (Barber y Odean, 2000).

    Args:
        traders (list[Agent]): Lista de agentes de la simulación.
        trades_df (pd.DataFrame): Registro histórico de transacciones.
        daily_values_gross (np.ndarray): Matriz de valores brutos diarios por agente.
        daily_values_net (np.ndarray): Matriz de valores netos diarios por agente.

    Returns:
        tuple:
            - float: Coeficiente Beta de turnover sobre el retorno bruto.
            - float: Coeficiente Beta de turnover sobre el retorno neto.
            - pd.DataFrame: DataFrame de datos transversales utilizados en la regresión OLS.
    """
    n_traders = len(traders)
    
    # --- 1. CÁLCULO DE RETORNOS TOTALES POR AGENTE ---
    returns_gross = (daily_values_gross[-1, :] - daily_values_gross[0, :]) / daily_values_gross[0, :]
    returns_net = (daily_values_net[-1, :] - daily_values_net[0, :]) / daily_values_net[0, :]
    
    turnover_list, initial_sizes, n_final_positions, portfolio_volatilities = [], [], [], []
    
    # --- 2. CONSTRUCCIÓN DE VARIABLES DE CONTROL TRANSVERSALES ---
    for i in range(n_traders):
        trader_trades = trades_df[trades_df['agent_id'] == i] if not trades_df.empty else pd.DataFrame()
        total_volume = (trader_trades['quantity'] * trader_trades['execution_price']).sum() if not trader_trades.empty else 0.0
        portfolio_size = traders[i].initial_portfolio_value
        
        turnover_list.append(total_volume / portfolio_size)
        initial_sizes.append(portfolio_size)
        n_final_positions.append(len(traders[i].portfolio))
        
        daily_rets = np.diff(daily_values_gross[:, i]) / daily_values_gross[:-1, i]
        portfolio_volatilities.append(np.std(daily_rets))
        
    df_reg = pd.DataFrame({
        'return_gross': returns_gross,
        'return_net': returns_net,
        'turnover': turnover_list,
        'portfolio_size': initial_sizes,
        'num_positions': n_final_positions,
        'risk_exposure': portfolio_volatilities,
        'const': 1.0
    })
    
    # --- 3. ESTIMACIÓN DE REGRESIONES OLS (BRUTO Y NETO) ---
    X = df_reg[['const', 'turnover', 'portfolio_size', 'num_positions', 'risk_exposure']]
    
    beta_gross = sm.OLS(df_reg['return_gross'], X).fit().params['turnover']
    beta_net = sm.OLS(df_reg['return_net'], X).fit().params['turnover']
    
    return beta_gross, beta_net, df_reg