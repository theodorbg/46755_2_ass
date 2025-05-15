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

    # Format axes with larger font size
    plt.xlabel("Conditional Value at Risk (CVaR) [kEUR]", fontsize=22, fontweight='bold')
    plt.ylabel("Expected Profit [kEUR]", fontsize=22, fontweight='bold')
    plt.title("Risk-Return Trade-off (One-Price, α = 0.90)", fontsize=22, fontweight='bold')

    # Set detailed y-axis ticks
    y_min = min(risk_results['expected_profit'])
    y_max = max(risk_results['expected_profit'])
    y_range = y_max - y_min
    plt.gca().yaxis.set_major_locator(plt.LinearLocator(10))
    plt.gca().yaxis.set_minor_locator(plt.LinearLocator(20))

    # Format tick labels with more precision
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}'))
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.2f}'))
    
    # Increase tick label size
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
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

    plt.xlabel('Profit [EUR]', fontsize=22, fontweight='bold')
    plt.ylabel('Number of Scenarios', fontsize=22, fontweight='bold')
    plt.title('Profit Distribution for Different Risk Levels (One-Price)', fontsize=22, fontweight='bold')
    plt.legend(fontsize=22)
    plt.grid(True, alpha=0.3)
    
    # Set tick label sizes
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    plt.tight_layout()
    plt.savefig('part1/results/step4/figures/risk_profit_distributions.png', dpi=300, bbox_inches='tight')
    print('\nPlotted profit distribution and saved to part1/results/step4/figures/risk_profit_distributions.png')
    plt.close()

def plot_profit_distribution_kde(risk_results, beta):
    """Plot profit distribution for different risk levels using KDE curves"""
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    selected_betas = [0.0, 0.5, 1.0]
    colors = ['#2ca02c', '#1f77b4', '#d62728']  # Green, Blue, Red
    line_styles = ['-', '--', '-.']
    
    # Plot KDE curves for each beta value
    for i, beta in enumerate(selected_betas):
        idx = int(beta * 10)
        profits = list(risk_results['scenario_profits'][idx].values())
        profits_k = [p/1000 for p in profits]  # Convert to thousands for better readability
        
        # Create KDE curve
        from scipy import stats
        kde = stats.gaussian_kde(profits_k)
        x = np.linspace(min(profits_k), max(profits_k), 1000)
        axes.plot(x, kde(x), color=colors[i], linestyle=line_styles[i], 
                 linewidth=3, label=f'β={beta:.1f}')
        
        # Add small histogram below the curve for reference
        axes.hist(profits_k, bins=30, alpha=0.2, color=colors[i], density=True)
    
    # Add vertical lines for means
    vertical_positions = [0.9, 0.8, 0.7]  # Staggered vertical positions
    for i, beta in enumerate(selected_betas):
        idx = int(beta * 10)
        profits = list(risk_results['scenario_profits'][idx].values())
        mean_profit = np.mean(profits)/1000
        axes.axvline(mean_profit, color=colors[i], linestyle=line_styles[i], 
                   alpha=0.7, linewidth=2)
        
        # Place annotations at different vertical positions
        axes.text(mean_profit + 2, axes.get_ylim()[1]*vertical_positions[i], f'μ={mean_profit:.1f}',
                color=colors[i], fontsize=18, ha='left', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, pad=5))
    
    axes.set_xlabel('Profit [kEUR]', fontsize=22, fontweight='bold')
    axes.set_ylabel('Density', fontsize=22, fontweight='bold')
    axes.set_title('Profit Distribution for Different Risk Levels (One-Price)', 
                  fontsize=22, fontweight='bold')
    axes.legend(fontsize=22, loc='upper right')
    axes.grid(True, alpha=0.3)
    axes.tick_params(axis='both', which='major', labelsize=22)
    
    plt.tight_layout()
    plt.savefig('part1/results/step4/figures/risk_profit_distributions_kde.png', 
                dpi=300, bbox_inches='tight')
    print('\nPlotted profit distribution and saved to part1/results/step4/figures/risk_profit_distributions_kde.png')
    plt.close()


def plot_risk_return_tradeoff_two_price(two_price_risk_results):
    plt.figure(figsize=(10, 6))
    plt.plot(two_price_risk_results['cvar'], two_price_risk_results['expected_profit'], 
            'ro-', linewidth=2, markersize=6)

    # Format axes with larger font size
    plt.xlabel("Conditional Value at Risk (CVaR) [kEUR]", fontsize=22, fontweight='bold')
    plt.ylabel("Expected Profit [kEUR]", fontsize=22, fontweight='bold')
    plt.title("Risk-Return Trade-off (Two-Price, α = 0.90)", fontsize=22, fontweight='bold')

    # Set detailed y-axis ticks
    y_min = min(two_price_risk_results['expected_profit'])
    y_max = max(two_price_risk_results['expected_profit'])
    y_range = y_max - y_min
    plt.gca().yaxis.set_major_locator(plt.LinearLocator(10))
    plt.gca().yaxis.set_minor_locator(plt.LinearLocator(20))

    # Format tick labels with more precision and larger font
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}'))
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.2f}'))
    
    # Increase tick label size
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

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

    plt.xlabel('Profit [kEUR]', fontsize=22, fontweight='bold')
    plt.ylabel('Number of Scenarios', fontsize=22, fontweight='bold')
    plt.title('Profit Distribution for Different Risk Levels (Two-Price)', fontsize=22, fontweight='bold')
    plt.legend(fontsize=22)
    plt.grid(True, alpha=0.3)
    
    # Set tick label sizes
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    plt.tight_layout()
    plt.savefig('part1/results/step4/figures/risk_profit_distributions_two_price.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_profit_distribution_kde_two_price(two_price_risk_results, beta):
    """Plot profit distribution for different risk levels using KDE curves for two-price scheme"""
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    selected_betas = [0.0, 0.5, 1.0]
    colors = ['#2ca02c', '#1f77b4', '#d62728']  # Green, Blue, Red
    line_styles = ['-', '--', '-.']
    n_points = len(two_price_risk_results['beta'])
    
    # Plot KDE curves for each beta value
    for i, beta in enumerate(selected_betas):
        # Calculate correct index based on number of points
        idx = int((n_points - 1) * beta)
        profits = list(two_price_risk_results['scenario_profits'][idx].values())
        profits_k = [p/1000 for p in profits]  # Convert to thousands for better readability
        
        # Create KDE curve
        from scipy import stats
        kde = stats.gaussian_kde(profits_k)
        x = np.linspace(min(profits_k), max(profits_k), 1000)
        axes.plot(x, kde(x), color=colors[i], linestyle=line_styles[i], 
                 linewidth=3, label=f'β={beta:.1f}')
        
        # Add small histogram below the curve for reference
        axes.hist(profits_k, bins=30, alpha=0.2, color=colors[i], density=True)
    
    # Add vertical lines for means
    vertical_positions = [0.9, 0.8, 0.7]  # Staggered vertical positions
    for i, beta in enumerate(selected_betas):
        idx = int((n_points - 1) * beta)
        profits = list(two_price_risk_results['scenario_profits'][idx].values())
        mean_profit = np.mean(profits)/1000
        axes.axvline(mean_profit, color=colors[i], linestyle=line_styles[i], 
                   alpha=0.7, linewidth=2)
        
        # Place annotations at different vertical positions
        axes.text(mean_profit + 2, axes.get_ylim()[1]*vertical_positions[i], f'μ={mean_profit:.1f}',
                color=colors[i], fontsize=18, ha='left', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, pad=5))
    
    axes.set_xlabel('Profit [kEUR]', fontsize=22, fontweight='bold')
    axes.set_ylabel('Density', fontsize=22, fontweight='bold')
    axes.set_title('Profit Distribution for Different Risk Levels (Two-Price)', 
                  fontsize=22, fontweight='bold')
    axes.legend(fontsize=22, loc='upper right')
    axes.grid(True, alpha=0.3)
    axes.tick_params(axis='both', which='major', labelsize=22)
    
    plt.tight_layout()
    plt.savefig('part1/results/step4/figures/risk_profit_distributions_two_price_kde.png', 
                dpi=300, bbox_inches='tight')
    print('\nPlotted two-price profit distribution and saved to part1/results/step4/figures/risk_profit_distributions_two_price_kde.png')
    plt.close()

def plot_combined_risk_return_tradeoff(one_price_risk_results, two_price_risk_results):
    """
    Plot the risk-return tradeoff for both one-price and two-price balancing schemes
    in a single figure for direct comparison.
    
    Args:
        one_price_risk_results: Results dictionary from analyze_risk_return_tradeoff
        two_price_risk_results: Results dictionary from analyze_two_price_risk_return_tradeoff
    """
    plt.figure(figsize=(12, 8))
    
    # Get normalization factors (risk-neutral profits - beta=0)
    one_price_base = one_price_risk_results['expected_profit'][0]
    two_price_base = two_price_risk_results['expected_profit'][0]
    
    # Normalize the data
    one_price_norm_profit = [p/one_price_base for p in one_price_risk_results['expected_profit']]
    one_price_norm_cvar = [c/one_price_base for c in one_price_risk_results['cvar']]
    
    two_price_norm_profit = [p/two_price_base for p in two_price_risk_results['expected_profit']]
    two_price_norm_cvar = [c/two_price_base for c in two_price_risk_results['cvar']]
    
    # Plot normalized one-price efficient frontier
    plt.plot(one_price_norm_cvar, one_price_norm_profit, 
             'bo-', linewidth=2.5, markersize=8, label='One-Price')
    
    # Plot normalized two-price efficient frontier
    plt.plot(two_price_norm_cvar, two_price_norm_profit, 
             'ro-', linewidth=2.5, markersize=8, label='Two-Price')
    
    # Annotate beta values on selected points with adjusted positions
    beta_values = [0, 0.5, 1]
    
    # Custom annotation positions
    for beta in beta_values:
        # Find indices corresponding to these beta values
        one_idx = int(beta * (len(one_price_risk_results['beta'])-1))
        two_idx = int(beta * (len(two_price_risk_results['beta'])-1))
        
        # Add annotations with customized positions
        if beta == 0:
            # Move beta=0 annotation down for one-price
            plt.annotate(f'β={beta}', 
                        xy=(one_price_norm_cvar[one_idx], one_price_norm_profit[one_idx]),
                        xytext=(10, -25), textcoords='offset points',
                        color='blue', fontsize=18, fontweight='bold')
            
            # Move beta=0 annotation down for two-price (but not as far as one-price)
            plt.annotate(f'β={beta}', 
                        xy=(two_price_norm_cvar[two_idx], two_price_norm_profit[two_idx]),
                        xytext=(10, -10), textcoords='offset points',
                        color='red', fontsize=18, fontweight='bold')
        
        elif beta == 0.5:
            # Move beta=0.5 for one-price down and to the left
            plt.annotate(f'β={beta}', 
                        xy=(one_price_norm_cvar[one_idx], one_price_norm_profit[one_idx]),
                        xytext=(-40, -25), textcoords='offset points',
                        color='blue', fontsize=18, fontweight='bold')
            
            plt.annotate(f'β={beta}', 
                        xy=(two_price_norm_cvar[two_idx], two_price_norm_profit[two_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        color='red', fontsize=18, fontweight='bold')
        
        elif beta == 1:
            # Move beta=1 for one-price further down
            plt.annotate(f'β={beta}', 
                        xy=(one_price_norm_cvar[one_idx], one_price_norm_profit[one_idx]),
                        xytext=(-40, -50), textcoords='offset points',
                        color='blue', fontsize=18, fontweight='bold')
            
            plt.annotate(f'β={beta}', 
                        xy=(two_price_norm_cvar[two_idx], two_price_norm_profit[two_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        color='red', fontsize=18, fontweight='bold')
    
    # Format axes with larger font size
    plt.xlabel("Conditional Value at Risk (CVaR) [kEUR]", fontsize=22, fontweight='bold')
    plt.ylabel("Expected Profit [kEUR]", fontsize=22, fontweight='bold')
    plt.title("Normalized Risk-Return Trade-off Comparison (α = 0.90)", fontsize=22, fontweight='bold')
    
    # Format tick labels with more precision
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}'))
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.2f}'))
    
    # Increase tick label size
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    # Add grid for both major and minor ticks
    plt.grid(True, which='major', alpha=0.3, linestyle='--')
    plt.grid(True, which='minor', alpha=0.1, linestyle=':')
    
    # Add legend with large font size
    plt.legend(fontsize=22, loc='best')
    
    plt.tight_layout()
    plt.savefig('part1/results/step4/figures/combined_risk_return_tradeoff.png', 
                dpi=300, bbox_inches='tight')
    print('\nPlotted combined risk-return tradeoff and saved to part1/results/step4/figures/combined_risk_return_tradeoff.png')
    plt.close()

def plot_combined_risk_return_tradeoff_normalized(one_price_risk_results, two_price_risk_results):
    """
    Plot the normalized risk-return tradeoff for both one-price and two-price balancing schemes
    in a single figure for direct comparison.
    
    Args:
        one_price_risk_results: Results dictionary from analyze_risk_return_tradeoff
        two_price_risk_results: Results dictionary from analyze_two_price_risk_return_tradeoff
    """
    plt.figure(figsize=(12, 8))
    
    # Get normalization factors (risk-neutral profits - beta=0)
    one_price_base = one_price_risk_results['expected_profit'][0]
    two_price_base = two_price_risk_results['expected_profit'][0]
    
    # Normalize the data
    one_price_norm_profit = [p/one_price_base for p in one_price_risk_results['expected_profit']]
    one_price_norm_cvar = [c/one_price_base for c in one_price_risk_results['cvar']]
    
    two_price_norm_profit = [p/two_price_base for p in two_price_risk_results['expected_profit']]
    two_price_norm_cvar = [c/two_price_base for c in two_price_risk_results['cvar']]
    
    # Plot normalized one-price efficient frontier
    plt.plot(one_price_norm_cvar, one_price_norm_profit, 
             'bo-', linewidth=2.5, markersize=8, label='One-Price')
    
    # Plot normalized two-price efficient frontier
    plt.plot(two_price_norm_cvar, two_price_norm_profit, 
             'ro-', linewidth=2.5, markersize=8, label='Two-Price')
    
    # Annotate beta values on selected points with adjusted positions
    beta_values = [0, 0.5, 1]
    
    # Custom annotation positions
    for beta in beta_values:
        # Find indices corresponding to these beta values
        one_idx = int(beta * (len(one_price_risk_results['beta'])-1))
        two_idx = int(beta * (len(two_price_risk_results['beta'])-1))
        
        # Add annotations with customized positions
        if beta == 0:
            # Move beta=0 annotation down for one-price
            plt.annotate(f'β={beta}', 
                        xy=(one_price_norm_cvar[one_idx], one_price_norm_profit[one_idx]),
                        xytext=(10, -50), textcoords='offset points',
                        color='blue', fontsize=18, fontweight='bold')
            
            # Move beta=0 annotation down for two-price (but not as far as one-price)
            plt.annotate(f'β={beta}', 
                        xy=(two_price_norm_cvar[two_idx], two_price_norm_profit[two_idx]),
                        xytext=(10, -20), textcoords='offset points',
                        color='red', fontsize=18, fontweight='bold')
        
        elif beta == 0.5:
            # Move beta=0.5 for one-price down and to the left
            plt.annotate(f'β={beta}', 
                        xy=(one_price_norm_cvar[one_idx], one_price_norm_profit[one_idx]),
                        xytext=(-40, -25), textcoords='offset points',
                        color='blue', fontsize=18, fontweight='bold')
            
            plt.annotate(f'β={beta}', 
                        xy=(two_price_norm_cvar[two_idx], two_price_norm_profit[two_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        color='red', fontsize=18, fontweight='bold')
        
        elif beta == 1:
            # Move beta=1 for one-price further down
            plt.annotate(f'β={beta}', 
                        xy=(one_price_norm_cvar[one_idx], one_price_norm_profit[one_idx]),
                        xytext=(-40, -50), textcoords='offset points',
                        color='blue', fontsize=18, fontweight='bold')
            
            plt.annotate(f'β={beta}', 
                        xy=(two_price_norm_cvar[two_idx], two_price_norm_profit[two_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        color='red', fontsize=18, fontweight='bold')
    
    # Format axes with larger font size
    plt.xlabel("Normalized CVaR (Relative to Risk-Neutral Profit)", fontsize=22, fontweight='bold')
    plt.ylabel("Normalized Expected Profit", fontsize=22, fontweight='bold')
    plt.title("Normalized Risk-Return Trade-off Comparison (α = 0.90)", fontsize=22, fontweight='bold')
    
    # Add reference text showing base values
    plt.text(0.02, 0.02, 
             f"One-Price Base: {one_price_base/1000:.1f} kEUR\nTwo-Price Base: {two_price_base/1000:.1f} kEUR",
             transform=plt.gca().transAxes, fontsize=16, bbox=dict(facecolor='white', alpha=0.7))
    
    # Format tick labels to show as percentages
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    
    # Increase tick label size
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    # Add grid for both major and minor ticks
    plt.grid(True, which='major', alpha=0.3, linestyle='--')
    plt.grid(True, which='minor', alpha=0.1, linestyle=':')
    
    # Add legend with large font size
    plt.legend(fontsize=22, loc='best')
    
    # Add horizontal line at y=1.0 to show risk-neutral profit level
    # plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5)
    # plt.text(max(one_price_norm_cvar + two_price_norm_cvar)*0.95, 1.01, 
    #          "Risk-Neutral Profit", fontsize=14, ha='right')
    
    plt.tight_layout()
    plt.savefig('part1/results/step4/figures/normalized_risk_return_tradeoff.png', 
                dpi=300, bbox_inches='tight')
    print('\nPlotted normalized risk-return tradeoff and saved to part1/results/step4/figures/normalized_risk_return_tradeoff.png')
    plt.close()

def plot_profit_boxplots(risk_results, scheme_type="One-Price"):
    """
    Plot boxplots of profit distributions across different risk aversion levels.
    
    Args:
        risk_results: Dictionary containing risk analysis results
        scheme_type: String indicating the balancing scheme type ("One-Price" or "Two-Price")
    """
    plt.figure(figsize=(14, 8))
    
    # Prepare data structure for boxplot
    boxplot_data = []
    labels = []
    
    # Get data for each beta value
    for i, beta in enumerate(risk_results['beta']):
        # Extract profits for this beta value
        profits = list(risk_results['scenario_profits'][i].values())
        # Convert to thousands for better readability
        profits_k = [p/1000 for p in profits]
        boxplot_data.append(profits_k)
        labels.append(f'{beta:.1f}')
    
    # Create boxplot
    bp = plt.boxplot(boxplot_data, patch_artist=True, notch=True, showfliers=True)
    
    # Customize boxplot colors
    for box in bp['boxes']:
        if scheme_type == "One-Price":
            box.set(facecolor='#2E86C1', alpha=0.7)
        else:
            box.set(facecolor='#E67E22', alpha=0.7)
    
    # Add overlay line for expected profit
    expected_profits = [p/1000 for p in risk_results['expected_profit']]
    plt.plot(range(1, len(labels)+1), expected_profits, 'k-', linewidth=2, 
             marker='o', markersize=8, label='Expected Profit')
    
    # Add reference line for risk-neutral expected profit
    risk_neutral_profit = expected_profits[0]
    plt.axhline(y=risk_neutral_profit, color='r', linestyle='--', 
                label=f'Risk-Neutral Profit: {risk_neutral_profit:.1f} kEUR')
    
    # Format chart
    plt.xlabel('Risk Aversion (β)', fontsize=22, fontweight='bold')
    plt.ylabel('Profit (kEUR)', fontsize=22, fontweight='bold')
    plt.title(f'Profit Distribution by Risk Level ({scheme_type})', fontsize=22, fontweight='bold')
    plt.xticks(range(1, len(labels)+1), labels, fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=22)
    
    plt.tight_layout()
    filename = f'risk_profit_boxplots_{scheme_type.lower().replace("-", "_")}.png'
    plt.savefig(f'part1/results/step4/figures/{filename}', dpi=300, bbox_inches='tight')
    print(f'\nPlotted profit distribution boxplots for {scheme_type} and saved to part1/results/step4/figures/{filename}')
    plt.close()


def plot_combined_profit_boxplots(one_price_risk_results, two_price_risk_results):
    """
    Plot side-by-side boxplots comparing profit distributions for one-price and two-price schemes.
    
    Args:
        one_price_risk_results: Results dictionary from analyze_risk_return_tradeoff
        two_price_risk_results: Results dictionary from analyze_two_price_risk_return_tradeoff
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    
    # Selected beta values to display (reduce crowding)
    selected_betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Process one-price data
    one_price_data = []
    one_price_expected = []
    labels = []
    
    for beta in selected_betas:
        idx = min(int(beta * 10), len(one_price_risk_results['beta']) - 1)
        profits = list(one_price_risk_results['scenario_profits'][idx].values())
        profits_k = [p/1000 for p in profits]
        one_price_data.append(profits_k)
        one_price_expected.append(one_price_risk_results['expected_profit'][idx]/1000)
        labels.append(f'{beta:.1f}')
    
    # Process two-price data
    two_price_data = []
    two_price_expected = []
    
    for beta in selected_betas:
        idx = min(int(beta * 10), len(two_price_risk_results['beta']) - 1)
        profits = list(two_price_risk_results['scenario_profits'][idx].values())
        profits_k = [p/1000 for p in profits]
        two_price_data.append(profits_k)
        two_price_expected.append(two_price_risk_results['expected_profit'][idx]/1000)
    
    # Create one-price boxplot
    bp1 = ax1.boxplot(one_price_data, patch_artist=True, notch=True, showfliers=True)
    for box in bp1['boxes']:
        box.set(facecolor='#2E86C1', alpha=0.7)
    
    # Add expected profit line
    ax1.plot(range(1, len(labels)+1), one_price_expected, 'k-', linewidth=2,
            marker='o', markersize=8, label='Expected Profit')
    
    # Add reference line
    ax1.axhline(y=one_price_expected[0], color='r', linestyle='--',
                label=f'Risk-Neutral: {one_price_expected[0]:.1f} kEUR')
    
    # Create two-price boxplot
    bp2 = ax2.boxplot(two_price_data, patch_artist=True, notch=True, showfliers=True)
    for box in bp2['boxes']:
        box.set(facecolor='#E67E22', alpha=0.7)
    
    # Add expected profit line
    ax2.plot(range(1, len(labels)+1), two_price_expected, 'k-', linewidth=2,
            marker='o', markersize=8, label='Expected Profit')
    
    # Add reference line
    ax2.axhline(y=two_price_expected[0], color='r', linestyle='--',
                label=f'Risk-Neutral: {two_price_expected[0]:.1f} kEUR')
    
    # Format axes
    ax1.set_xlabel('Risk Aversion (β)', fontsize=22, fontweight='bold')
    ax1.set_ylabel('Profit (kEUR)', fontsize=22, fontweight='bold')
    ax1.set_title('One-Price Balancing Scheme', fontsize=22, fontweight='bold')
    ax1.set_xticks(range(1, len(labels)+1))
    ax1.set_xticklabels(labels, fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=20)
    
    ax2.set_xlabel('Risk Aversion (β)', fontsize=22, fontweight='bold')
    ax2.set_title('Two-Price Balancing Scheme', fontsize=22, fontweight='bold')
    ax2.set_xticks(range(1, len(labels)+1))
    ax2.set_xticklabels(labels, fontsize=22)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=20)
    
    fig.suptitle('Profit Distribution Comparison by Risk Level', fontsize=24, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('part1/results/step4/figures/combined_profit_boxplots.png', dpi=300, bbox_inches='tight')
    print('\nPlotted combined profit boxplots and saved to part1/results/step4/figures/combined_profit_boxplots.png')
    plt.close()

def plot_profit_violinplots(risk_results, scheme_type="One-Price"):
    """
    Plot violin plots of profit distributions across different risk aversion levels.
    
    Args:
        risk_results: Dictionary containing risk analysis results
        scheme_type: String indicating the balancing scheme type ("One-Price" or "Two-Price")
    """
    plt.figure(figsize=(14, 8))
    
    # Prepare data structure for violin plot
    violin_data = []
    labels = []
    
    # Get data for each beta value
    for i, beta in enumerate(risk_results['beta']):
        # Extract profits for this beta value
        profits = list(risk_results['scenario_profits'][i].values())
        # Convert to thousands for better readability
        profits_k = [p/1000 for p in profits]
        violin_data.append(profits_k)
        labels.append(f'{beta:.1f}')
    
    # Create violin plot
    vp = plt.violinplot(violin_data, showmeans=True, showmedians=True)
    
    # Customize violin colors
    if scheme_type == "One-Price":
        for pc in vp['bodies']:
            pc.set_facecolor('#2E86C1')
            pc.set_alpha(0.7)
    else:
        for pc in vp['bodies']:
            pc.set_facecolor('#E67E22')
            pc.set_alpha(0.7)
    
    # Customize other elements
    vp['cmeans'].set_color('black')
    vp['cmeans'].set_linewidth(2)
    vp['cmedians'].set_color('darkblue')
    vp['cmedians'].set_linewidth(2)
    
    # Add overlay line for expected profit
    expected_profits = [p/1000 for p in risk_results['expected_profit']]
    plt.plot(range(1, len(labels)+1), expected_profits, 'ko-', linewidth=2, 
             marker='o', markersize=8, label='Expected Profit')
    
    # Add reference line for risk-neutral expected profit
    risk_neutral_profit = expected_profits[0]
    plt.axhline(y=risk_neutral_profit, color='r', linestyle='--', 
                label=f'Risk-Neutral Profit: {risk_neutral_profit:.1f} kEUR')
    
    # Format chart
    plt.xlabel('Risk Aversion (β)', fontsize=22, fontweight='bold')
    plt.ylabel('Profit (kEUR)', fontsize=22, fontweight='bold')
    plt.title(f'Profit Distribution by Risk Level ({scheme_type})', fontsize=22, fontweight='bold')
    plt.xticks(range(1, len(labels)+1), labels, fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=22)
    
    plt.tight_layout()
    filename = f'risk_profit_violinplots_{scheme_type.lower().replace("-", "_")}.png'
    plt.savefig(f'part1/results/step4/figures/{filename}', dpi=300, bbox_inches='tight')
    print(f'\nPlotted profit distribution violin plots for {scheme_type} and saved to part1/results/step4/figures/{filename}')
    plt.close()


def plot_combined_profit_violinplots(one_price_risk_results, two_price_risk_results):
    """
    Plot side-by-side violin plots comparing profit distributions for one-price and two-price schemes.
    
    Args:
        one_price_risk_results: Results dictionary from analyze_risk_return_tradeoff
        two_price_risk_results: Results dictionary from analyze_two_price_risk_return_tradeoff
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    
    # Selected beta values to display (reduce crowding)
    selected_betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Process one-price data
    one_price_data = []
    one_price_expected = []
    labels = []
    
    for beta in selected_betas:
        idx = min(int(beta * 10), len(one_price_risk_results['beta']) - 1)
        profits = list(one_price_risk_results['scenario_profits'][idx].values())
        profits_k = [p/1000 for p in profits]
        one_price_data.append(profits_k)
        one_price_expected.append(one_price_risk_results['expected_profit'][idx]/1000)
        labels.append(f'{beta:.1f}')
    
    # Process two-price data
    two_price_data = []
    two_price_expected = []
    
    for beta in selected_betas:
        idx = min(int(beta * 10), len(two_price_risk_results['beta']) - 1)
        profits = list(two_price_risk_results['scenario_profits'][idx].values())
        profits_k = [p/1000 for p in profits]
        two_price_data.append(profits_k)
        two_price_expected.append(two_price_risk_results['expected_profit'][idx]/1000)
    
    # Create one-price violin plot
    vp1 = ax1.violinplot(one_price_data, showmeans=True, showmedians=True)
    for pc in vp1['bodies']:
        pc.set_facecolor('#2E86C1')
        pc.set_alpha(0.7)
    vp1['cmeans'].set_color('black')
    vp1['cmeans'].set_linewidth(2)
    vp1['cmedians'].set_color('darkblue')
    vp1['cmedians'].set_linewidth(2)
    
    # Add expected profit line
    ax1.plot(range(1, len(labels)+1), one_price_expected, 'ko-', linewidth=2,
            marker='o', markersize=8, label='Expected Profit')
    
    # Add reference line
    ax1.axhline(y=one_price_expected[0], color='r', linestyle='--',
                label=f'Risk-Neutral: {one_price_expected[0]:.1f} kEUR')
    
    # Create two-price violin plot
    vp2 = ax2.violinplot(two_price_data, showmeans=True, showmedians=True)
    for pc in vp2['bodies']:
        pc.set_facecolor('#E67E22')
        pc.set_alpha(0.7)
    vp2['cmeans'].set_color('black')
    vp2['cmeans'].set_linewidth(2)
    vp2['cmedians'].set_color('darkblue')
    vp2['cmedians'].set_linewidth(2)
    
    # Add expected profit line
    ax2.plot(range(1, len(labels)+1), two_price_expected, 'ko-', linewidth=2,
            marker='o', markersize=8, label='Expected Profit')
    
    # Add reference line
    ax2.axhline(y=two_price_expected[0], color='r', linestyle='--',
                label=f'Risk-Neutral: {two_price_expected[0]:.1f} kEUR')
    
    # Format axes
    ax1.set_xlabel('Risk Aversion (β)', fontsize=22, fontweight='bold')
    ax1.set_ylabel('Profit (kEUR)', fontsize=22, fontweight='bold')
    ax1.set_title('One-Price Balancing Scheme', fontsize=22, fontweight='bold')
    ax1.set_xticks(range(1, len(labels)+1))
    ax1.set_xticklabels(labels, fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=20)
    
    ax2.set_xlabel('Risk Aversion (β)', fontsize=22, fontweight='bold')
    ax2.set_title('Two-Price Balancing Scheme', fontsize=22, fontweight='bold')
    ax2.set_xticks(range(1, len(labels)+1))
    ax2.set_xticklabels(labels, fontsize=22)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=20)
    
    fig.suptitle('Profit Distribution Comparison by Risk Level', fontsize=24, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('part1/results/step4/figures/combined_profit_violinplots.png', dpi=300, bbox_inches='tight')
    print('\nPlotted combined profit violin plots and saved to part1/results/step4/figures/combined_profit_violinplots.png')
    plt.close()