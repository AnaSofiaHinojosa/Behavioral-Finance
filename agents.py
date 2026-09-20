import numpy as np

class Agent:
    """Representa a un inversionista individual con sesgos comportamentales inyectados."""

    def __init__(self, agent_id, alpha_disp, alpha_over, initial_cash, n_assets, n_initial_positions=None):
        """Inicializa un agente trader y construye su portafolio inicial.

        Args:
            agent_id (int): Identificador único del agente.
            alpha_disp (float): Magnitud inyectada del sesgo Disposition Effect [0, 1].
            alpha_over (float): Magnitud inyectada del sesgo Overconfidence [0, 1].
            initial_cash (float): Capital inicial en USD otorgado al agente.
            n_assets (int): Universo total de activos disponibles en el mercado.
            n_initial_positions (int, optional): Cantidad de posiciones iniciales a abrir.
                Si es None, se muestrea uniformemente entre 5 y 30.
        """
        self.id = agent_id
        self.alpha_disp = alpha_disp
        self.alpha_over = alpha_over
        self.initial_cash = initial_cash
        
        # --- 1. CUENTAS INDEPENDIENTES DE CAJA (GROSS Y NET) ---
        self.cash_net = initial_cash
        self.cash_gross = initial_cash
        
        # --- 2. SELECCIÓN ALEATORIA DE POSICIONES INICIALES ---
        if n_initial_positions is None:
            self.n_initial_positions = int(np.random.uniform(5, 31))
        else:
            self.n_initial_positions = n_initial_positions
            
        self.portfolio = {} # Estructura: {asset_id: {'quantity': q, 'buy_price': p}}
        
        selected_assets = np.random.choice(n_assets, size=self.n_initial_positions, replace=False)
        alloc_per_asset = (initial_cash * 0.7) / self.n_initial_positions
        
        for asset in selected_assets:
            p0 = 100.0  # Precio base de inicialización
            qty = alloc_per_asset / p0
            self.portfolio[asset] = {'quantity': qty, 'buy_price': p0}
            self.cash_net -= alloc_per_asset
            self.cash_gross -= alloc_per_asset
            
        self.initial_portfolio_value = initial_cash

    def decide_trades(self, day, current_prices, spread_bps=10.0, commission_usd=1.0, confound=None):
        """Evalúa el portafolio actual y ejecuta órdenes de compra/venta según sesgos o confusores.

        Args:
            day (int): Día actual de la simulación.
            current_prices (pd.Series): Precios de mercado de todos los activos en el día t.
            spread_bps (float, optional): Spread Bid-Ask en puntos básicos. Por defecto 10.0.
            commission_usd (float, optional): Comisión fija por transacción. Por defecto 1.0.
            confound (str, optional): Regla alternativa de venta ('rebalancing' o 'belief_in_reversal').

        Returns:
            list[dict]: Lista de diccionarios con las transacciones ejecutadas en el día.
        """
        trades_executed = []
        base_sell_prob = 0.05
        
        # --- 1. DECISIONES DE VENTA (DISPOSICIÓN Y CONFUSORES) ---
        for asset, pos_data in list(self.portfolio.items()):
            cur_price = current_prices.iloc[asset] if isinstance(asset, (int, np.integer)) else current_prices.loc[asset]
            buy_price = pos_data['buy_price']
            qty = pos_data['quantity']
            
            sell_prob = base_sell_prob
            
            if confound == 'rebalancing':
                # Regla Racional: Vender si la posición ganó más del 10% para rebalancear
                if cur_price > buy_price * 1.10:
                    sell_prob = 0.80
                else:
                    sell_prob = 0.02
            elif confound == 'belief_in_reversal':
                # Creencia Racional: Vender lo que subió por expectativa de reversión a la media
                if cur_price > buy_price:
                    sell_prob = base_sell_prob * 3.0
                else:
                    sell_prob = base_sell_prob * 0.2
            else:
                # Mecanismo Estándar de Disposición (Hazard Rate)
                if cur_price > buy_price:
                    sell_prob = base_sell_prob * (1.0 + 2.0 * self.alpha_disp)
                elif cur_price < buy_price:
                    sell_prob = base_sell_prob * max(0.01, (1.0 - 0.8 * self.alpha_disp))

            # Ejecución de la Venta
            if np.random.uniform(0, 1) < sell_prob:
                half_spread = (spread_bps / 10000.0) / 2.0
                p_net_sell = cur_price * (1.0 - half_spread)
                p_gross_sell = cur_price
                
                self.cash_net += (qty * p_net_sell) - commission_usd
                self.cash_gross += (qty * p_gross_sell)
                
                trades_executed.append({
                    'day': day,
                    'agent_id': self.id,
                    'type': 'SELL',
                    'asset': asset,
                    'quantity': qty,
                    'execution_price': p_net_sell,
                    'buy_price': buy_price
                })
                del self.portfolio[asset]

        # --- 2. DECISIONES DE COMPRA (SOBRECONFIANZA / TURNOVER) ---
        buy_prob = 0.02 + 0.20 * self.alpha_over
        if np.random.uniform(0, 1) < buy_prob:
            available_assets = [a for a in range(len(current_prices)) if a not in self.portfolio]
            if available_assets and self.cash_net > 1000:
                asset_to_buy = np.random.choice(available_assets)
                cur_price = current_prices.iloc[asset_to_buy] if isinstance(asset_to_buy, (int, np.integer)) else current_prices.loc[asset_to_buy]
                
                buy_amount = min(self.cash_net * 0.2, 10000)
                half_spread = (spread_bps / 10000.0) / 2.0
                p_net_buy = cur_price * (1.0 + half_spread)
                p_gross_buy = cur_price
                
                qty = (buy_amount - commission_usd) / p_net_buy if buy_amount > commission_usd else 0
                
                if qty > 0:
                    self.cash_net -= (qty * p_net_buy + commission_usd)
                    self.cash_gross -= (qty * p_gross_buy)
                    self.portfolio[asset_to_buy] = {'quantity': qty, 'buy_price': cur_price}
                    
                    trades_executed.append({
                        'day': day,
                        'agent_id': self.id,
                        'type': 'BUY',
                        'asset': asset_to_buy,
                        'quantity': qty,
                        'execution_price': p_net_buy,
                        'buy_price': cur_price
                    })

        return trades_executed