import numpy as np

class Agent:
    def __init__(self, agent_id, alpha_disp, alpha_over, initial_cash, n_assets, n_initial_positions=None):
        self.id = agent_id
        self.alpha_disp = alpha_disp
        self.alpha_over = alpha_over
        self.initial_cash = initial_cash
        
        # Dos cuentas de caja independientes para separar Gross y Net
        self.cash_net = initial_cash
        self.cash_gross = initial_cash
        
        # n_initial_positions Muestreado entre 5 y 30 si no se especifica
        if n_initial_positions is None:
            self.n_initial_positions = int(np.random.uniform(5, 31))
        else:
            self.n_initial_positions = n_initial_positions
            
        self.portfolio = {} # {asset_id: {'quantity': q, 'buy_price': p}}
        
        # Inicialización de posiciones iniciales
        selected_assets = np.random.choice(n_assets, size=self.n_initial_positions, replace=False)
        alloc_per_asset = (initial_cash * 0.7) / self.n_initial_positions
        
        for asset in selected_assets:
            p0 = 100.0 # Precio base de inicialización
            qty = alloc_per_asset / p0
            self.portfolio[asset] = {'quantity': qty, 'buy_price': p0}
            self.cash_net -= alloc_per_asset
            self.cash_gross -= alloc_per_asset
            
        self.initial_portfolio_value = initial_cash

    def decide_trades(self, day, current_prices, spread_bps=10.0, commission_usd=1.0, confound=None):
        trades_executed = []
        base_sell_prob = 0.05
        
        # --- 1. DECISIONES DE VENTA (DISPOSICIÓN Y CONFUSORES) ---
        for asset, pos_data in list(self.portfolio.items()):
            cur_price = current_prices[asset]
            buy_price = pos_data['buy_price']
            qty = pos_data['quantity']
            
            sell_prob = base_sell_prob
            
            if confound == 'rebalancing':
                # Regla de rebalanceo: Vender si la posición creció más del 10%
                if cur_price > buy_price * 1.10:
                    sell_prob = 0.80
                else:
                    sell_prob = 0.02
            elif confound == 'belief_in_reversal':
                # Reversión a la media: Vender lo que subió pensando que caerá
                if cur_price > buy_price:
                    sell_prob = base_sell_prob * 3.0
                else:
                    sell_prob = base_sell_prob * 0.2
            else:
                # Mecanismo Estándar de Disposición (Hazard Rate)
                if cur_price > buy_price: # Ganancia en papel
                    sell_prob = base_sell_prob * (1.0 + 2.0 * self.alpha_disp)
                elif cur_price < buy_price: # Pérdida en papel
                    sell_prob = base_sell_prob * max(0.01, (1.0 - 0.8 * self.alpha_disp))

            if np.random.uniform(0, 1) < sell_prob:
                # Precios de Ejecución
                half_spread = (spread_bps / 10000.0) / 2.0
                p_net_sell = cur_price * (1.0 - half_spread) # Involucra spread
                p_gross_sell = cur_price                     # Precio medio puro
                
                # Actualización de Cajas Independientes
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
                cur_price = current_prices[asset_to_buy]
                
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