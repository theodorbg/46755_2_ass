import gurobipy as gp
from gurobipy import GRB
import numpy as np


# %% Risk-Averse one-price offering strategy

def solve_risk_averse_one_price(in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS, alpha=0.90, beta=0):
    """
    Solve the risk-averse optimization problem for one-price balancing scheme.
    """
    model = gp.Model("WindFarmRiskAverse")
        
    # Decision variables
    p_DA = model.addVars(N_HOURS, lb=0, ub=CAPACITY_WIND_FARM, name="p_DA")
    profit = model.addVars(in_sample_scenarios.keys(), name="profit")
    zeta = model.addVar(name="zeta")  # VaR variable
    eta = model.addVars(in_sample_scenarios.keys(), lb=0, name="eta")  # CVaR auxiliary variables
    
    # Constants
    N_SCENARIOS = len(in_sample_scenarios)
    SCENARIO_PROB = 1.0 / N_SCENARIOS
    
    # Calculate profit for each scenario
    for s in in_sample_scenarios:
        scenario = in_sample_scenarios[s]
        profit_expr = gp.LinExpr(0)
        
        for h in range(N_HOURS):
            wind_actual = scenario.loc[h, 'wind']
            price_DA = scenario.loc[h, 'price']
            price_BAL = scenario.loc[h, 'balancing_price']
            
            profit_expr.addTerms(price_DA, p_DA[h])
            profit_expr.addConstant(price_BAL * wind_actual)
            profit_expr.addTerms(-price_BAL, p_DA[h])
        
        model.addConstr(profit[s] == profit_expr, f"profit_scenario_{s}")
        # CVaR constraint following the mathematical formulation
        model.addConstr(eta[s] >= zeta - profit[s], f"cvar_scenario_{s}")
    
    # Expected profit and CVaR terms
    expected_profit = gp.quicksum(SCENARIO_PROB * profit[s] for s in in_sample_scenarios)
    cvar_term = zeta - (1 / (1 - alpha)) * gp.quicksum(SCENARIO_PROB * eta[s] for s in in_sample_scenarios)
    
    # Objective following the mathematical formulation
    model.setObjective((1 - beta) * expected_profit + beta * cvar_term, GRB.MAXIMIZE)
    
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Model not solved optimally. Status: {model.status}")
    
    # Extract results
    optimal_offers = [p_DA[h].X for h in range(N_HOURS)]
    scenario_profits = {s: profit[s].X for s in in_sample_scenarios}
    exp_profit_value = expected_profit.getValue()
    cvar_value = cvar_term.getValue()
    
    return optimal_offers, exp_profit_value, cvar_value, scenario_profits


def analyze_risk_return_tradeoff(in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS, beta_values=None):
    """
    Solve risk-averse model across multiple beta values and return risk-return results.

    Returns:
        results: dict with expected_profit, CVaR, optimal_offers, scenario_profits for each beta
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
        offers, exp_profit, cvar, scen_profits = solve_risk_averse_one_price(
            in_sample_scenarios,
            CAPACITY_WIND_FARM,
            N_HOURS,
            alpha=0.90,
            beta=beta
        )
        results['expected_profit'].append(exp_profit)
        results['cvar'].append(cvar)
        results['optimal_offers'].append(offers)
        results['scenario_profits'].append(scen_profits)

    return results

# %% Risk-Averse two-price offering strategy

def solve_risk_averse_two_price(in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS, alpha=0.90, beta=0):
    """
    Solve the risk-averse optimization problem for two-price balancing scheme.
    """
    # Create model with logging disabled
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        model = gp.Model("WindFarmRiskAverseTwoPrice", env=env)
    
    # Set parameters silently
    model.setParam('DualReductions', 0)
    
    # Decision variables with tighter bounds
    p_DA = model.addVars(N_HOURS, lb=0, ub=CAPACITY_WIND_FARM, name="p_DA")
    pos_imbalance = model.addVars(in_sample_scenarios.keys(), N_HOURS, lb=0, 
                                 ub=CAPACITY_WIND_FARM, name="pos_imbalance")
    neg_imbalance = model.addVars(in_sample_scenarios.keys(), N_HOURS, lb=0, 
                                 ub=CAPACITY_WIND_FARM, name="neg_imbalance")
    profit = model.addVars(in_sample_scenarios.keys(), name="profit")
    zeta = model.addVar(lb=-GRB.INFINITY, name="zeta")  # VaR variable
    eta = model.addVars(in_sample_scenarios.keys(), lb=0, name="eta")
    
    # Constants
    N_SCENARIOS = len(in_sample_scenarios)
    SCENARIO_PROB = 1.0 / N_SCENARIOS
    
# Calculate profit for each scenario with bounds
    for s in in_sample_scenarios:
        scenario = in_sample_scenarios[s]
        profit_expr = gp.LinExpr(0)
        
        for h in range(N_HOURS):
            wind_actual = scenario.loc[h, 'wind']
            price_DA = scenario.loc[h, 'price']
            price_BAL = scenario.loc[h, 'balancing_price']
            condition = scenario.loc[h, 'condition']
            
            # Two-price balancing scheme pricing
            if condition == 0:  # System excess
                surplus_price = min(price_BAL, price_DA)  # Ensure lower price
                deficit_price = price_DA
            else:  # System deficit
                surplus_price = price_DA
                deficit_price = max(price_BAL, price_DA)  # Ensure higher price
            
            # Add bounded profit components
            profit_expr.addTerms(price_DA, p_DA[h])
            profit_expr.addTerms(surplus_price, pos_imbalance[s, h])
            profit_expr.addTerms(-deficit_price, neg_imbalance[s, h])
            
            # Imbalance constraints with bounds
            model.addConstr(
                wind_actual - p_DA[h] == pos_imbalance[s, h] - neg_imbalance[s, h],
                f"imbalance_{s}_{h}"
            )
            model.addConstr(
                pos_imbalance[s, h] + neg_imbalance[s, h] <= CAPACITY_WIND_FARM,
                f"total_imbalance_{s}_{h}"
            )
        
        model.addConstr(profit[s] == profit_expr, f"profit_scenario_{s}")
        model.addConstr(eta[s] >= zeta - profit[s], f"cvar_scenario_{s}")
    
    # Expected profit and CVaR terms
    expected_profit = gp.quicksum(SCENARIO_PROB * profit[s] for s in in_sample_scenarios)
    cvar_term = zeta - (1 / (1 - alpha)) * gp.quicksum(SCENARIO_PROB * eta[s] for s in in_sample_scenarios)
    
    # Risk-averse objective
    objective = (1 - beta) * expected_profit + beta * cvar_term
    model.setObjective(objective, GRB.MAXIMIZE)
    
    # Optimize
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Model not solved optimally. Status: {model.status}")
    
    # Extract results
    optimal_offers = [p_DA[h].X for h in range(N_HOURS)]
    scenario_profits = {s: profit[s].X for s in in_sample_scenarios}
    exp_profit_value = expected_profit.getValue()
    cvar_value = cvar_term.getValue()
    
    return optimal_offers, exp_profit_value, cvar_value, scenario_profits

def analyze_two_price_risk_return_tradeoff(in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS, beta_values=None):
    """
    Analyze risk-return tradeoff for two-price balancing scheme.
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
        optimal_offers, exp_profit, cvar, scen_profits = solve_risk_averse_two_price(
            in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS, beta=beta
        )
        
        results['expected_profit'].append(exp_profit)
        results['cvar'].append(cvar)
        results['optimal_offers'].append(optimal_offers)
        results['scenario_profits'].append(scen_profits)
    
    return results