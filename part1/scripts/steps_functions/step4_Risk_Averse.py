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

def plot_risk_return_tradeoff(risk_results):
    plt.figure(figsize=(10, 6))
    plt.plot(risk_results['cvar'], risk_results['expected_profit'], 'bo-', linewidth=2, markersize=6)

    # Format axes
    plt.xlabel("Conditional Value at Risk (CVaR) [kEUR]")
    plt.ylabel("Expected Profit [kEUR]")
    plt.title("Risk-Return Trade-off (One-Price, α = 0.90)")

    # Set detailed y-axis ticks
    y_min = min(risk_results['expected_profit'])
    y_max = max(risk_results['expected_profit'])
    y_range = y_max - y_min
    plt.gca().yaxis.set_major_locator(plt.LinearLocator(10))  # Reduced number of ticks
    plt.gca().yaxis.set_minor_locator(plt.LinearLocator(20))  # Reduced number of minor ticks

    # Format tick labels with more precision
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}'))
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.2f}'))
    # Add grid for both major and minor ticks
    plt.grid(True, which='major', alpha=0.3, linestyle='--')
    plt.grid(True, which='minor', alpha=0.1, linestyle=':')

    plt.tight_layout()
    plt.savefig('part1/results/step4/figures/risk_return_tradeoff.png', dpi=300, bbox_inches='tight')
    print('\nPlotted risk return results and saved to part1/results/step4/figures/risk_return_tradeoff.png')

    plt.close()

def plot_profit_distribution(scenario_profits, beta):
    plt.figure(figsize=(12, 6))
    selected_betas = [0.0, 0.5, 1.0]
    for beta in selected_betas:
        idx = int(beta * 10)
        profits = list(risk_results['scenario_profits'][idx].values())
        plt.hist(profits, bins=30, alpha=0.5, label=f'β={beta:.1f}')

    plt.xlabel('Profit [EUR]')
    plt.ylabel('Number of Scenarios')
    plt.title('Profit Distribution for Different Risk Levels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('part1/results/step4/figures/risk_profit_distributions.png', dpi=300, bbox_inches='tight')
    print('\nPlotted profit distribution and saved to part1/results/step4/figures/risk_profit_distributions.png')

    plt.close()

def plot_risk_return_tradeoff_two_price(risk_results):
    plt.figure(figsize=(10, 6))
    plt.plot(two_price_risk_results['cvar'], two_price_risk_results['expected_profit'], 
            'ro-', linewidth=2, markersize=6)

    # Format axes
    plt.xlabel("Conditional Value at Risk (CVaR) [kEUR]")
    plt.ylabel("Expected Profit [kEUR]")
    plt.title("Risk-Return Trade-off (Two-Price, α = 0.90)")

    # Set detailed y-axis ticks
    y_min = min(two_price_risk_results['expected_profit'])
    y_max = max(two_price_risk_results['expected_profit'])
    y_range = y_max - y_min
    plt.gca().yaxis.set_major_locator(plt.LinearLocator(10))  # Reduced number of ticks
    plt.gca().yaxis.set_minor_locator(plt.LinearLocator(20))  # Minor ticks

    # Format tick labels with more precision
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}'))
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.2f}'))

    # Add grid for both major and minor ticks
    plt.grid(True, which='major', alpha=0.3, linestyle='--')
    plt.grid(True, which='minor', alpha=0.1, linestyle=':')

    plt.tight_layout()
    plt.savefig('part1/results/step4/figures/risk_return_tradeoff_two_price.png', dpi=300, bbox_inches='tight')
    print('\nPlotted two-price risk return results and saved to part1/results/step4/figures/risk_return_tradeoff_two_price.png')
    plt.close()

def plot_profit_distribution_two_price(scenario_profits, beta):
    plt.figure(figsize=(12, 6))
    selected_betas = [0.0, 0.5, 1.0]
    n_points = len(two_price_risk_results['beta'])

    for beta in selected_betas:
        # Calculate correct index based on number of points
        idx = int((n_points - 1) * beta)
        profits = list(two_price_risk_results['scenario_profits'][idx].values())
        plt.hist(profits, bins=30, alpha=0.5, label=f'β={beta:.1f}')

    plt.xlabel('Profit [kEUR]')
    plt.ylabel('Number of Scenarios')
    plt.title('Profit Distribution for Different Risk Levels (Two-Price)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('part1/results/step4/figures/risk_profit_distributions_two_price.png', dpi=300, bbox_inches='tight')
    plt.close()