import numpy as np

class Agent:
    def __init__(self, agent_id, alpha_disp, alpha_over, initial_cash, n_assets, n_initial_positions=10):
        self.id = agent_id
        self.alpha_disp = alpha_disp
        self.alpha_over = alpha_over
        self.cash = initial_cash
        self.initial_portfolio_value = initial_cash
        self.portfolio = {}
        
        # Asignación de portafolio inicial
        chosen_assets = np.random.choice(n_assets, size=n_initial_positions, replace=False)
        allocation_per_asset = (initial_cash * 0.7) / n_initial_positions
        
        for asset_idx in chosen_assets:
            asset_name = f"Asset_{asset_idx}"
            qty = allocation_per_asset / 100.0
            self.portfolio[asset_name] = {'quantity': qty, 'buy_price': 100.0}
            
        self.cash -= allocation_per_asset * n_initial_positions

    def decide_trades(self, day, current_prices, spread_bps=10.0, commission_usd=1.0, confound=None):
        trades_today = []

        # --- A. REGLA DE VENTA (DISPOSITION) ---
        positions_to_sell = []
        for asset, data in self.portfolio.items():
            price_today = current_prices[asset]
            buy_price = data['buy_price']
            is_gain = price_today > buy_price
            
            base_prob = 0.05 + 0.15 * self.alpha_over
            
            if confound == "rebalancing":
                sell_prob = 0.30 if is_gain else 0.05
            elif confound == "belief_in_reversal":
                sell_prob = 0.25 if is_gain else 0.02
            else:
                if is_gain:
                    sell_prob = base_prob * (1.0 + 2.0 * self.alpha_disp)
                else:
                    sell_prob = base_prob * (1.0 - 0.8 * self.alpha_disp)
            
            sell_prob = np.clip(sell_prob, 0.01, 0.95)
            if np.random.rand() < sell_prob:
                positions_to_sell.append(asset)

        for asset in positions_to_sell:
            data = self.portfolio[asset]
            raw_price = current_prices[asset]
            bid_price = raw_price * (1.0 - (spread_bps / 10000.0) / 2)
            
            revenue = (data['quantity'] * bid_price) - commission_usd
            self.cash += revenue
            
            trades_today.append({
                'day': day, 'agent_id': self.id, 'asset': asset, 'type': 'SELL',
                'quantity': data['quantity'], 'execution_price': bid_price,
                'buy_price': data['buy_price'], 'gross_price': raw_price,
                'commission': commission_usd
            })
            del self.portfolio[asset]

        # --- B. REGLA DE COMPRA (OVERCONFIDENCE / TURNOVER) ---
        buy_prob = 0.02 + 0.20 * self.alpha_over
        if np.random.rand() < buy_prob and self.cash > 2000:
            available_assets = [a for a in current_prices.index if a not in self.portfolio]
            if available_assets:
                target_asset = np.random.choice(available_assets)
                raw_price = current_prices[target_asset]
                ask_price = raw_price * (1.0 + (spread_bps / 10000.0) / 2)
                
                investment = min(self.cash * 0.4, 10000)
                qty = (investment - commission_usd) / ask_price
                
                if qty > 0:
                    self.cash -= investment
                    self.portfolio[target_asset] = {'quantity': qty, 'buy_price': ask_price}
                    trades_today.append({
                        'day': day, 'agent_id': self.id, 'asset': target_asset, 'type': 'BUY',
                        'quantity': qty, 'execution_price': ask_price,
                        'buy_price': ask_price, 'gross_price': raw_price,
                        'commission': commission_usd
                    })

        return trades_today

    def evaluate_portfolio(self, current_prices):
        position_val = sum(data['quantity'] * current_prices[asset] for asset, data in self.portfolio.items())
        return self.cash + position_val