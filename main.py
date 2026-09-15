import numpy as np
import pandas as pd
from market import generate_market_data
from simulation import run_scenario
from estimators import estimate_disposition_effect, estimate_overconfidence
from bootstrap import run_cluster_bootstrap
from config import RANDOM_SEED, N_TRADERS, N_BOOTSTRAP

def main():
    np.random.seed(RANDOM_SEED)
    print("--- 1. GENERANDO PRECIOS DEL MERCADO ---")
    market_prices = generate_market_data(seed=RANDOM_SEED)
    
    scenarios_config = {
        1: {'name': 'Null', 'alpha_disp': 0.0, 'alpha_over': 0.0, 'confound': None},
        2: {'name': 'Disposition Low', 'alpha_disp': 0.3, 'alpha_over': 0.0, 'confound': None},
        3: {'name': 'Disposition High', 'alpha_disp': 0.8, 'alpha_over': 0.0, 'confound': None},
        4: {'name': 'Turnover Low', 'alpha_disp': 0.0, 'alpha_over': 0.3, 'confound': None},
        5: {'name': 'Turnover High', 'alpha_disp': 0.0, 'alpha_over': 0.8, 'confound': None},
        6: {'name': 'Both Active', 'alpha_disp': 0.8, 'alpha_over': 0.8, 'confound': None},
        7: {'name': 'Rebalancing Confound', 'alpha_disp': 0.0, 'alpha_over': 0.0, 'confound': 'rebalancing'},
        8: {'name': 'Mean-Reversion Confound', 'alpha_disp': 0.0, 'alpha_over': 0.0, 'confound': 'belief_in_reversal'}
    }

    results = []
    print("\n--- 2. EJECUTANDO ESCENARIOS (1 al 8) ---")
    for sc_id, cfg in scenarios_config.items():
        print(f"Corriendo Escenario {sc_id}: {cfg['name']}...")
        traders, trades_df, daily_values_gross = run_scenario(cfg, market_prices, n_traders=N_TRADERS)
        
        disp_metrics = estimate_disposition_effect(trades_df, market_prices)
        beta_over, _ = estimate_overconfidence(traders, trades_df, daily_values_gross)
        
        se_disp, se_over = run_cluster_bootstrap(traders, trades_df, market_prices, daily_values_gross, n_boot=N_BOOTSTRAP)
        
        results.append({
            'Scenario': sc_id,
            'Name': cfg['name'],
            'α_disp': cfg['alpha_disp'],
            'α_over': cfg['alpha_over'],
            'PGR-PLR': round(disp_metrics['diff'], 4),
            'SE (Disp)': round(se_disp, 4),
            'β_Turnover': round(beta_over, 4),
            'SE (Overconf)': round(se_over, 4)
        })

    summary_df = pd.DataFrame(results)
    print("\n================ RESULTADOS FINALES ================")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()