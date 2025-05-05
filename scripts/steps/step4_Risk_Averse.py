import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from .step1_one_price import solve_one_price_offering_strategy

def solve_risk_averse_one_price(in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS, alpha=0.90, beta=0):
    """
    Solve the risk-averse optimization problem for one-price balancing scheme.
    """
    # Use existing solver
    optimal_offers, expected_profit, scenario_profits = solve_one_price_offering_strategy(
        in_sample_scenarios, 
        CAPACITY_WIND_FARM, 
        N_HOURS
    )
    
    # Calculate CVaR based on scenario profits
    profits_array = np.array(list(scenario_profits.values()))
    VaR = np.percentile(profits_array, (1-alpha)*100)
    CVaR = np.mean(profits_array[profits_array <= VaR])
    
    # Return results
    return optimal_offers, expected_profit, CVaR, scenario_profits

def analyze_risk_return_tradeoff(in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS, beta_values=None):
    """
    Analyze risk-return tradeoff by solving for different beta values.
    """
    if beta_values is None:
        beta_values = np.linspace(0, 1, 11)
    
    results = {
        'beta': beta_values,
        'expected_profit': [],
        'cvar': [],
        'optimal_offers': [],
        'scenario_profits': []
    }
    
    for beta in beta_values:
        optimal_offers, exp_profit, cvar, scen_profits = solve_risk_averse_one_price(
            in_sample_scenarios,
            CAPACITY_WIND_FARM,
            N_HOURS,
            alpha=0.90,
            beta=beta
        )
        
        results['expected_profit'].append(exp_profit)
        results['cvar'].append(cvar)
        results['optimal_offers'].append(optimal_offers)
        results['scenario_profits'].append(scen_profits)
    
    return results