import random
from gurobipy import Model, GRB
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import os

def solve_step1(profiles_matrix, num_profiles, num_minutes, N, epsilon,
                consumption_lb=0, consumption_ub=600, BETA_UB=0, ZETA_UB=0):
    # Model setup
    model = Model("CVaR_P90_SlideFormulation") # Initialize a Gurobi model named "CVaR_P90_SlideFormulation".
    model.setParam('OutputFlag', 0) # Suppress Gurobi's console output during optimization.

    # Decision variables
    # c: The reserve capacity bid (upward flexibility). This is what we want to maximize.
    # It represents the amount of power we commit to be able to reduce consumption by.
    # Bounds are [0, 600] kW, reflecting the overall consumption range.
    c = model.addVar(lb=consumption_lb, ub=consumption_ub, name="reserve_bid")           # c↑ (upward reserve)
    # beta: The Value at Risk (VaR) at the (1-epsilon) confidence level.
    # It represents the threshold for "bad" outcomes (shortfalls).
    # For CVaR, beta itself is a variable to be determined by the model.
    # It's constrained to be non-positive (ub=0) as per the slide formulation for maximizing 'c'.
    beta = model.addVar(ub=BETA_UB, name="beta")                       # β ≤ 0
    # zeta: Auxiliary variables representing the shortfall beyond beta for each profile and minute.
    # zeta_m,ω = max(0, (c - consumption_m,ω) - beta)
    # These are non-negative (lb=0).
    zeta = model.addVars(num_profiles, num_minutes, lb=ZETA_UB, name="zeta")  # ζ_m,ω ≥ 0

    # Objective: maximize the reserve bid 'c'
    model.setObjective(c, GRB.MAXIMIZE) # Set the objective to maximize the variable 'c'.

    # Constraints:
    # These constraints define the relationship between 'c', 'beta', 'zeta', and the consumption profiles.
    for i in range(num_profiles):  # Iterate over each consumption profile
        for t in range(num_minutes): # Iterate over each minute in the profile
            # Shortfall definition constraint:
            # (c - consumption_m,ω) represents the potential shortfall if consumption is consumption_m,ω
            # and we bid 'c'.
            # zeta_m,ω must be at least this shortfall.
            # This effectively linearizes zeta_m,ω >= c - profiles_matrix[i, t]
            # (Note: The formulation in slides might be slightly different for minimization problems,
            # here it's adapted for maximizing 'c' with beta <= 0)
            # This constraint ensures zeta captures the positive part of (c - consumption)
            model.addConstr(c - profiles_matrix[i, t] <= zeta[i, t], name=f"shortfall_{i}_{t}")

            # Beta bound constraint:
            # This constraint, along with the shortfall definition, helps define zeta correctly
            # in relation to beta. It ensures that zeta_i,t is at least beta.
            # Combined with zeta_i,t >= c - profile_matrix[i,t] and zeta_i,t >= 0,
            # and the CVaR constraint, it correctly models the CVaR.
            # For the slide formulation where beta <= 0, this means zeta_i,t >= beta.
            model.addConstr(beta <= zeta[i, t], name=f"beta_bound_{i}_{t}")

    # CVaR constraint:
    # The average of the shortfalls (zeta values) that exceed beta should be related to beta.
    # (1/N) * sum(zeta_i,t) is the expected shortfall given that the shortfall is greater than beta.
    # This constraint links the average of these "tail" shortfalls (zeta) to beta.
    # The formulation (1/N) * sum(zeta) <= (1-epsilon) * beta is a bit unusual.
    # A more standard CVaR formulation for minimizing cost (where beta > 0) is:
    # beta + (1/epsilon) * sum(zeta_i,t) / N <= CVaR_target or objective.
    # For maximizing 'c' with beta <= 0, the constraint structure might be specific to the
    # "slide formulation" being referenced.
    # This constraint essentially states that the average of all zeta values
    # (which are max(0, shortfall - beta) effectively, due to other constraints and objective)
    # must be less than or equal to (1-epsilon) times beta.
    # Given beta <= 0, (1-epsilon)*beta will also be <=0.
    # This implies the sum of zetas (which are >=0) must be very small or zero.
    # This specific formulation needs to be carefully checked against the source "slide formulation".
    # A common CVaR (minimizing losses L) is: beta + (1/epsilonN) * sum(max(0, L_i - beta))
    # If we are maximizing 'c', and shortfall is S = c - P, then we want to limit the CVaR of S.
    # Let L = -(c - P). We want to maximize c s.t. CVaR(P-c) is controlled or VaR(P-c) is controlled.
    # The current formulation seems to be a direct application of a specific CVaR variant.
    model.addConstr(
        (1 / N) * sum(zeta[i, t] for i in range(num_profiles) for t in range(num_minutes)) # Average of all zeta values
        <= (1 - epsilon) * beta, # Must be less than or equal to (1-epsilon) * beta
        name="cvar_constraint"
    )

    # Solve the model
    model.optimize()

    return c.X, beta.X, zeta

def solve_milp_model(profiles_matrix, num_profiles, num_minutes, max_violations, M,
                     consumption_lb=0, consumption_ub=600, BETA_UB=0, ZETA_UB=0):
    # Create MILP model
    model = Model("ALSOX_MILP")
    model.setParam('OutputFlag', 0)

    # Variables
    c = model.addVar(lb=consumption_lb, ub=consumption_ub, name="reserve_bid")  # c↑
    y = model.addVars(num_profiles, num_minutes, vtype=GRB.BINARY, name="y")  # y_{m,ω}

    # Objective: Maximize reserve bid
    model.setObjective(c, GRB.MAXIMIZE)

    # Constraints
    for i in range(num_profiles):
        for t in range(num_minutes):
            model.addConstr(c - profiles_matrix[i, t] <= M * y[i, t], f"viol_{i}_{t}")

    # Total violation constraint
    model.addConstr(sum(y[i, t] for i in range(num_profiles) for t in range(num_minutes)) <= max_violations, "violation_budget")

    # Solve
    model.optimize()

    return c.X

def prepare_profile_data(profiles, max_power=600):
    """
    Convert consumption profiles into a numpy array suitable for plotting.
    
    Args:
        profiles (list): List of ConsumptionProfile objects or arrays
        max_power (float): Maximum power value for the y-axis
        
    Returns:
        tuple: (profiles_matrix, num_profiles, num_minutes, power_axis)
    """
    # Extract matrix
    profiles_matrix = np.array([p.profile if hasattr(p, 'profile') else p for p in profiles])
    num_profiles, num_minutes = profiles_matrix.shape

    # Create power axis
    power_axis = np.linspace(0, max_power, 200)
    
    return profiles_matrix, num_profiles, num_minutes, power_axis

def calculate_probability_matrix(profiles_matrix, power_axis):
    """
    Calculate the probability matrix for the heatmap.
    Calculate the P90 line (10th percentile of consumption).
    
    Args:
        profiles_matrix (numpy.ndarray): Matrix of consumption profiles
        power_axis (numpy.ndarray): Array of power values
        
    Returns:
        numpy.ndarray: Probability matrix
        numpy.ndarray: P90 line values
    """
    num_profiles, num_minutes = profiles_matrix.shape
    prob_matrix = np.zeros((len(power_axis), num_minutes))
    
    for t in range(num_minutes):
        for i, p in enumerate(power_axis):
            prob_matrix[i, t] = np.mean(profiles_matrix[:, t] >= p)
    
    p90_line = np.percentile(profiles_matrix, 10, axis=0)
            
    return prob_matrix, p90_line
    
def plot_flexibility_analysis(
    profiles_matrix, 
    prob_matrix,
    p90_line, 
    r_cvar, 
    r_alsox,
    power_axis,
    num_minutes,
    title="Upwards Flexibility $F^\\uparrow$ with P90 and Reserve Bids",
    show_plot=True
):
    """
    Create a combined plot showing flexibility analysis with reserve bids.
    
    Args:
        profiles_matrix (numpy.ndarray): Matrix of consumption profiles
        prob_matrix (numpy.ndarray): Probability matrix for heatmap
        p90_line (numpy.ndarray): P90 values for each minute
        r_cvar (float): CVaR reserve bid value
        r_alsox (float): ALSO-X reserve bid value
        power_axis (numpy.ndarray): Array of power values
        num_minutes (int): Number of minutes
        show_plot (bool): Whether to show the plot
    """
    title = "Upwards Flexibility $F^\\uparrow$ with P90 and Reserve Bids"

    plt.figure(figsize=(12, 5))
    extent = [0, num_minutes, power_axis[0], power_axis[-1]]
    
    # Create heatmap
    plt.imshow(prob_matrix[::-1, :], aspect='auto', extent=extent, cmap='inferno', vmin=0, vmax=1)
    plt.colorbar(label='Probability of Available Flexibility')
    
    # Plot mean consumption
    plt.plot(np.arange(num_minutes), profiles_matrix.mean(axis=0), 
             color='lime', label='Mean Consumption', linewidth=2.5)
    
    # Plot P90 line
    plt.plot(np.arange(num_minutes), p90_line, 
             color='white', linestyle='--', linewidth=2.5, label='P90 Level')
    
    # Plot reserve bids
    plt.hlines(r_cvar, 0, num_minutes, colors='cyan', linestyles='--', 
               linewidth=2.5, label=f'CVaR Reserve ({r_cvar:.0f} kW)')
    plt.hlines(r_alsox, 0, num_minutes, colors='deepskyblue', linestyles='-.', 
               linewidth=2.5, label=f'ALSO-X Reserve ({r_alsox:.0f} kW)')
    
    # Add labels and formatting
    plt.title(title, fontsize=16)
    plt.xlabel("Time of Hour [min]", fontsize=14)
    plt.ylabel("Power [kW]", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    
    save_path = 'part2/results/step1/figures/flexibility_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # Show or close
    if show_plot:
        plt.show()
    else:
        plt.close()

def create_flexibility_plot(profiles, r_cvar, r_alsox, max_power=600, show_plot=True):
    """
    Convenience function that combines all steps to create a flexibility plot.
    
    Args:
        profiles (list): List of ConsumptionProfile objects
        r_cvar (float): CVaR reserve bid value
        r_alsox (float): ALSO-X reserve bid value
        max_power (float): Maximum power value
        title (str): Plot title
        save_path (str, optional): Path to save the plot
        show_plot (bool): Whether to show the plot
    """
    # Prepare data
    profiles_matrix, num_profiles, num_minutes, power_axis = prepare_profile_data(
        profiles, max_power)
    
    # Calculate probability matrix and P90 line
    prob_matrix, p90_line = calculate_probability_matrix(profiles_matrix, power_axis)
    
    # Create plot
    plot_flexibility_analysis(
        profiles_matrix, prob_matrix, p90_line, r_cvar, r_alsox,
        power_axis, num_minutes, show_plot = False
        )
    
    return profiles_matrix, prob_matrix, p90_line