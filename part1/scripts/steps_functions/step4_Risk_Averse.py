import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt


# %% Risk-Averse one-price offering strategy

def solve_risk_averse_one_price(in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS, alpha=0.90, beta=0):
    """
    Solve the risk-averse optimization problem for one-price balancing scheme.
    
    This function formulates and solves a stochastic optimization model that balances expected profit
    and risk (measured by Conditional Value-at-Risk) for a wind power producer participating in 
    electricity markets under a one-price balancing scheme.
    
    Args:
        in_sample_scenarios: Dictionary of scenarios for model training
        CAPACITY_WIND_FARM: Maximum production capacity of wind farm (MW)
        N_HOURS: Number of hours in planning horizon (typically 24)
        alpha: Confidence level for CVaR calculation (default: 0.90)
        beta: Risk-aversion parameter (0=risk-neutral, 1=most risk-averse)
        
    Returns:
        tuple: (optimal_offers, expected_profit, cvar_value, scenario_profits)
    """
    # --- Setup Gurobi environment with output suppression ---
    # Creating a clean environment and setting OutputFlag=0 prevents Gurobi
    # from printing detailed solver information to console
    with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)  # Suppress all Gurobi output
            env.start()
            model = gp.Model("WindFarmRiskAverse", env=env)        
        
    # --- Decision variables ---
    # p_DA[h]: Day-ahead market power bid for hour h (MW)
    p_DA = model.addVars(N_HOURS, lb=0, ub=CAPACITY_WIND_FARM, name="p_DA")
    
    # profit[s]: Total profit for scenario s (€)
    profit = model.addVars(in_sample_scenarios.keys(), name="profit")
    
    # zeta: Value-at-Risk (VaR) at confidence level alpha
    # This is the threshold value for the tail losses we're concerned about
    zeta = model.addVar(name="zeta")  # VaR variable
    
    # eta[s]: Auxiliary variables for CVaR calculation
    # These represent the shortfall below VaR in each scenario (if any)
    eta = model.addVars(in_sample_scenarios.keys(), lb=0, name="eta")  # CVaR auxiliary variables
    
    # --- Model parameters ---
    N_SCENARIOS = len(in_sample_scenarios)  # Total number of scenarios
    SCENARIO_PROB = 1.0 / N_SCENARIOS       # Equal probability for each scenario
    
    # --- Calculate profit and set up CVaR constraints for each scenario ---
    for s in in_sample_scenarios:
        scenario = in_sample_scenarios[s]  # Get scenario data
        profit_expr = gp.LinExpr(0)        # Initialize profit expression
        
        # Iterate through each hour to calculate hourly profit components
        for h in range(N_HOURS):
            # Extract scenario data for this hour
            wind_actual = scenario.loc[h, 'wind']           # Realized wind production (MW)
            price_DA = scenario.loc[h, 'price']            # Day-ahead market price (€/MWh)
            price_BAL = scenario.loc[h, 'balancing_price']  # Balancing market price (€/MWh)
            
            # --- Profit calculation for one-price balancing scheme ---
            # Component 1: Revenue from day-ahead market
            profit_expr.addTerms(price_DA, p_DA[h])  # p_DA[h] * price_DA
            
            # Component 2: Revenue/cost from balancing market
            profit_expr.addConstant(price_BAL * wind_actual)  # Constant term: price_BAL * wind_actual
            profit_expr.addTerms(-price_BAL, p_DA[h])         # Variable term: -price_BAL * p_DA[h]
            
            # The complete profit formula for hour h in one-price scheme is:
            # profit_h = p_DA[h] * price_DA + (wind_actual - p_DA[h]) * price_BAL
            # which simplifies to:
            # profit_h = p_DA[h] * (price_DA - price_BAL) + wind_actual * price_BAL
        
        # Add constraint defining total profit for this scenario
        model.addConstr(profit[s] == profit_expr, f"profit_scenario_{s}")
        
        # --- CVaR constraints ---
        # This constraint captures the shortfall below VaR for the scenario
        # eta[s] must be at least (zeta - profit[s]) or zero, whichever is greater
        model.addConstr(eta[s] >= zeta - profit[s], f"cvar_scenario_{s}")
    
    # --- Expected profit calculation ---
    # Sum of scenario profits weighted by their probability
    expected_profit = gp.quicksum(SCENARIO_PROB * profit[s] for s in in_sample_scenarios)
    
    # --- CVaR calculation ---
    # CVaR = VaR - (1/(1-alpha)) * E[max(0, VaR - profit)]
    # VaR is represented by zeta
    # E[max(0, VaR - profit)] is the expected shortfall below VaR, represented by eta variables
    cvar_term = zeta - (1 / (1 - alpha)) * gp.quicksum(SCENARIO_PROB * eta[s] for s in in_sample_scenarios)
    
    # --- Objective function: Risk-return trade-off ---
    # beta controls the trade-off between expected profit and risk:
    # - beta=0: Risk-neutral (maximize expected profit only)
    # - beta=1: Most risk-averse (maximize CVaR only)
    # - 0<beta<1: Balance between expected profit and CVaR
    model.setObjective((1 - beta) * expected_profit + beta * cvar_term, GRB.MAXIMIZE)
    
    # --- Solve the model ---
    model.optimize()
    
    # --- Check solution status ---
    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Model not solved optimally. Status: {model.status}")
    
    # --- Extract and return results ---
    optimal_offers = [p_DA[h].X for h in range(N_HOURS)]  # Optimal hourly bids
    scenario_profits = {s: profit[s].X for s in in_sample_scenarios}  # Profit in each scenario
    exp_profit_value = expected_profit.getValue()  # Expected profit value
    cvar_value = cvar_term.getValue()  # CVaR value
    
    return optimal_offers, exp_profit_value, cvar_value, scenario_profits


def analyze_risk_return_tradeoff(in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS, beta_values=None):
    """
    Solve risk-averse model across multiple beta values and return risk-return results.
    
    This function performs a sensitivity analysis by varying the risk-aversion parameter (beta)
    to generate the complete risk-return efficient frontier. For each beta value, it solves
    the optimization model and collects the resulting expected profit and CVaR metrics.
    
    Args:
        in_sample_scenarios: Dictionary of scenarios for model training
        CAPACITY_WIND_FARM: Maximum production capacity of wind farm (MW)
        N_HOURS: Number of hours in planning horizon (typically 24)
        beta_values: Optional list of beta values to analyze; default is 11 values from 0 to 1
        
    Returns:
        results: Dictionary containing arrays of beta values and corresponding metrics:
                - beta: The risk-aversion parameters tested
                - expected_profit: Expected profit for each beta
                - cvar: Conditional Value-at-Risk for each beta
                - optimal_offers: Optimal hourly offers for each beta
                - scenario_profits: Scenario-specific profits for each beta
    """
    # --- Default beta values setup ---
    # If no beta values are provided, create 11 equally spaced values from 0 to 1
    # Beta=0 is risk-neutral (maximize expected profit only)
    # Beta=1 is fully risk-averse (maximize CVaR only)
    if beta_values is None:
        beta_values = np.linspace(0, 1, 11)  # [0.0, 0.1, 0.2, ..., 1.0]

    # --- Initialize results dictionary ---
    # We'll collect all metrics for each beta value to allow complete analysis
    results = {
        'beta': beta_values,         # Risk-aversion parameters
        'expected_profit': [],       # Expected profit values
        'cvar': [],                  # Conditional Value-at-Risk values
        'optimal_offers': [],        # Optimal day-ahead market offers
        'scenario_profits': []       # Profit for each scenario
    }

    # --- Loop through each beta value ---
    # For each beta, solve the risk-averse optimization model and collect results
    for beta in beta_values:
        # Solve the risk-averse model with the current beta value
        offers, exp_profit, cvar, scen_profits = solve_risk_averse_one_price(
            in_sample_scenarios,     # Training scenarios
            CAPACITY_WIND_FARM,      # Wind farm capacity
            N_HOURS,                 # Planning horizon
            alpha=0.90,              # Fixed confidence level (90%)
            beta=beta                # Current risk-aversion parameter
        )
        
        # Store results for this beta value
        results['expected_profit'].append(exp_profit)  # Expected profit
        results['cvar'].append(cvar)                   # CVaR value
        results['optimal_offers'].append(offers)       # Optimal hourly offers
        results['scenario_profits'].append(scen_profits)  # Scenario profits
        
        # Note: As beta increases:
        # - Expected profit typically decreases
        # - CVaR typically increases (less negative)
        # - Optimal offers become more conservative
        # This demonstrates the fundamental trade-off between risk and return

    # Return the complete results dictionary for further analysis
    return results

# %% Risk-Averse two-price offering strategy

def solve_risk_averse_two_price(in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS, alpha=0.90, beta=0):
    """
    Solve the risk-averse optimization problem for two-price balancing scheme.
    
    This function implements a stochastic optimization model for a wind power producer
    participating in a two-price balancing market. In two-price schemes, surplus and
    deficit imbalances are settled at different prices depending on the overall
    system condition (excess or deficit).
    
    Args:
        in_sample_scenarios: Dictionary of training scenarios
        CAPACITY_WIND_FARM: Maximum wind farm capacity (MW)
        N_HOURS: Planning horizon length (typically 24 hours)
        alpha: Confidence level for CVaR calculation (default: 0.90)
        beta: Risk-aversion parameter (0=risk-neutral, 1=most risk-averse)
        
    Returns:
        tuple: (optimal_offers, expected_profit, cvar_value, scenario_profits)
    """
    # --- Setup Gurobi environment with output suppression ---
    # Creating a clean environment and disabling output messages
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)  # Suppress solver output messages
        env.start()  # Initialize the environment
        model = gp.Model("WindFarmRiskAverseTwoPrice", env=env)  # Create model with this environment
    
    # --- Model parameter settings ---
    # DualReductions=0 helps avoid numerical issues with unbounded dual variables
    model.setParam('DualReductions', 0)  # Disables presolve dual reductions
    
    # --- Decision variables ---
    # p_DA: Day-ahead market power bid for each hour
    p_DA = model.addVars(N_HOURS, lb=0, ub=CAPACITY_WIND_FARM, name="p_DA")
    
    # pos_imbalance: Positive imbalance (surplus) for each scenario and hour
    # When actual production exceeds day-ahead bid: wind_actual > p_DA
    pos_imbalance = model.addVars(in_sample_scenarios.keys(), N_HOURS, lb=0, 
                                 ub=CAPACITY_WIND_FARM, name="pos_imbalance")
    
    # neg_imbalance: Negative imbalance (deficit) for each scenario and hour
    # When actual production is less than day-ahead bid: wind_actual < p_DA
    neg_imbalance = model.addVars(in_sample_scenarios.keys(), N_HOURS, lb=0, 
                                 ub=CAPACITY_WIND_FARM, name="neg_imbalance")
    
    # profit: Total profit for each scenario
    profit = model.addVars(in_sample_scenarios.keys(), name="profit")
    
    # zeta: Value-at-Risk (VaR) at confidence level alpha
    # Unlike in one-price scheme, profits could potentially be very negative,
    # so we allow zeta to take negative values
    zeta = model.addVar(lb=-GRB.INFINITY, name="zeta")  # VaR variable without lower bound
    
    # eta: Auxiliary variables for CVaR calculation
    eta = model.addVars(in_sample_scenarios.keys(), lb=0, name="eta")
    
    # --- Model parameters ---
    N_SCENARIOS = len(in_sample_scenarios)  # Total number of scenarios
    SCENARIO_PROB = 1.0 / N_SCENARIOS       # Equal probability for each scenario
    
    # --- Calculate profit for each scenario with imbalance handling ---
    for s in in_sample_scenarios:
        scenario = in_sample_scenarios[s]  # Get data for this scenario
        profit_expr = gp.LinExpr(0)        # Initialize profit expression
        
        # Process each hour in the planning horizon
        for h in range(N_HOURS):
            # Get scenario data for this hour
            wind_actual = scenario.loc[h, 'wind']           # Actual wind production (MW)
            price_DA = scenario.loc[h, 'price']             # Day-ahead price (€/MWh)
            price_BAL = scenario.loc[h, 'balancing_price']  # Balancing price (€/MWh)
            condition = scenario.loc[h, 'condition']        # System condition (0=excess, 1=deficit)
            
            # --- Two-price balancing scheme pricing logic ---
            # In two-price schemes, the price depends on both:
            # 1. Whether the producer has surplus or deficit
            # 2. Whether the system as a whole is in excess or deficit
            if condition == 0:  # System excess condition
                # System has excess power (oversupply)
                # Surplus is penalized (producer receives less than day-ahead price)
                # Deficit is not penalized (producer pays day-ahead price)
                surplus_price = min(price_BAL, price_DA)  # Lower of the two prices
                deficit_price = price_DA                  # Standard day-ahead price
            else:  # System deficit condition
                # System has deficit of power (undersupply)
                # Surplus is not rewarded (producer receives day-ahead price)
                # Deficit is penalized (producer pays more than day-ahead price)
                surplus_price = price_DA                   # Standard day-ahead price
                deficit_price = max(price_BAL, price_DA)   # Higher of the two prices
            
            # --- Build profit expression for this hour ---
            # Component 1: Revenue from day-ahead market
            profit_expr.addTerms(price_DA, p_DA[h])  # p_DA[h] * price_DA
            
            # Component 2: Revenue from positive imbalance (surplus)
            profit_expr.addTerms(surplus_price, pos_imbalance[s, h])
            
            # Component 3: Cost of negative imbalance (deficit)
            profit_expr.addTerms(-deficit_price, neg_imbalance[s, h])
            
            # --- Imbalance constraints ---
            # Constraint 1: Ensure imbalance variables correctly represent the difference
            # between actual wind production and day-ahead bid
            # wind_actual - p_DA[h] = pos_imbalance - neg_imbalance
            model.addConstr(
                wind_actual - p_DA[h] == pos_imbalance[s, h] - neg_imbalance[s, h],
                f"imbalance_{s}_{h}"
            )
            
            # Constraint 2: Ensure the sum of positive and negative imbalances
            # doesn't exceed physical capacity (this is a logical bound)
            model.addConstr(
                pos_imbalance[s, h] + neg_imbalance[s, h] <= CAPACITY_WIND_FARM,
                f"total_imbalance_{s}_{h}"
            )
        
        # --- Define total profit for this scenario ---
        model.addConstr(profit[s] == profit_expr, f"profit_scenario_{s}")
        
        # --- CVaR constraint for this scenario ---
        # eta[s] must be at least the shortfall below VaR (if any)
        model.addConstr(eta[s] >= zeta - profit[s], f"cvar_scenario_{s}")
    
    # --- Risk-averse objective components ---
    # Expected profit: probability-weighted sum of all scenario profits
    expected_profit = gp.quicksum(SCENARIO_PROB * profit[s] for s in in_sample_scenarios)
    
    # CVaR term: VaR minus the expected shortfall beyond VaR, scaled by 1/(1-alpha)
    # CVaR represents the expected profit in the worst (1-alpha)% of scenarios
    cvar_term = zeta - (1 / (1 - alpha)) * gp.quicksum(SCENARIO_PROB * eta[s] for s in in_sample_scenarios)
    
    # --- Combined objective with risk-return tradeoff ---
    # Beta controls the balance between expected profit and CVaR
    # Higher beta values prioritize risk reduction over expected profit
    objective = (1 - beta) * expected_profit + beta * cvar_term
    model.setObjective(objective, GRB.MAXIMIZE)
    
    # --- Solve the optimization model ---
    model.optimize()
    
    # --- Check solution status ---
    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Model not solved optimally. Status: {model.status}")
    
    # --- Extract and return results ---
    # Optimal day-ahead bids for each hour
    optimal_offers = [p_DA[h].X for h in range(N_HOURS)]
    
    # Profit for each scenario under the optimal strategy
    scenario_profits = {s: profit[s].X for s in in_sample_scenarios}
    
    # Expected profit value across all scenarios
    exp_profit_value = expected_profit.getValue()
    
    # CVaR value (expected profit in worst-case scenarios)
    cvar_value = cvar_term.getValue()
    
    return optimal_offers, exp_profit_value, cvar_value, scenario_profits

def analyze_two_price_risk_return_tradeoff(in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS, beta_values=None):
    """
    Analyze risk-return tradeoff for two-price balancing scheme.
    
    This function performs a sensitivity analysis across different risk-aversion levels
    to generate the efficient frontier for the two-price balancing scheme. For each beta
    value, it solves the corresponding optimization problem and collects performance metrics.
    
    Args:
        in_sample_scenarios: Dictionary of scenarios for model training
        CAPACITY_WIND_FARM: Maximum wind farm capacity in MW
        N_HOURS: Number of hours in planning horizon
        beta_values: Optional list of beta values to analyze (default: 11 values from 0 to 1)
        
    Returns:
        dict: Results containing beta values and corresponding performance metrics
    """
    # --- User feedback on process start ---
    # Inform the user that the analysis has begun
    print('Analyzing risk-return tradeoff for two-price balancing scheme...')

    # --- Default beta values setup ---
    # If no specific beta values are provided, create a linear spacing
    # from 0 (risk-neutral) to 1 (fully risk-averse)
    if beta_values is None:
        beta_values = np.linspace(0, 1, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
    
    # --- Initialize results container ---
    # Create dictionary to store all metrics for each beta value
    results = {
        'beta': beta_values,         # The risk-aversion parameters tested
        'expected_profit': [],       # Expected profit for each beta
        'cvar': [],                  # Conditional Value-at-Risk for each beta
        'optimal_offers': [],        # Optimal day-ahead bids for each beta
        'scenario_profits': []       # Scenario-specific profits for each beta
    }
    
    # --- Perform analysis for each beta value ---
    # Loop through risk-aversion levels from risk-neutral to fully risk-averse
    for beta in beta_values:
        # Solve the two-price risk-averse model with the current beta
        optimal_offers, exp_profit, cvar, scen_profits = solve_risk_averse_two_price(
            in_sample_scenarios,     # Training scenarios
            CAPACITY_WIND_FARM,      # Wind farm capacity
            N_HOURS,                 # Planning horizon
            beta=beta                # Current risk-aversion parameter (alpha=0.9 by default)
        )
        
        # --- Store results for this beta value ---
        results['expected_profit'].append(exp_profit)     # Store expected profit
        results['cvar'].append(cvar)                      # Store CVaR value
        results['optimal_offers'].append(optimal_offers)  # Store optimal hourly bids
        results['scenario_profits'].append(scen_profits)  # Store scenario profits
    
    # --- Provide completion feedback ---
    # Verify results were successfully generated and inform user
    if results['expected_profit']:  # Check if we have any results
        print('Risk-return tradeoff analysis completed successfully.')
    else:
        # Error handling: alert user if no results were generated
        print('No results found. Please check the input data and parameters.')
    
    # Return the complete results dictionary for visualization or further analysis
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

def plot_profit_distribution(risk_results, beta):
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

def plot_risk_return_tradeoff_two_price(two_price_risk_results):
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

def plot_profit_distribution_two_price(two_price_risk_results, beta):
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