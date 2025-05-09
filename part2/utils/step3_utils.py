import random
from gurobipy import Model, GRB
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import os

def prepare_data_matrices(in_sample_profiles, out_sample_profiles):
    """
    Prepare data matrices from the consumption profiles.
    
    Parameters:
    - in_sample_profiles: List of in-sample ConsumptionProfile objects.
    - out_sample_profiles: List of out-of-sample ConsumptionProfile objects.
    
    Returns:
    - in_matrix: 2D NumPy array for in-sample profiles.
    - out_matrix: 2D NumPy array for out-of-sample profiles.
    """
    # Convert list of ConsumptionProfile objects into a 2D NumPy array.
    # Each row is a profile, each column is a minute's consumption.
    in_matrix = np.array([
        p.profile if hasattr(p, 'profile') else p for p in in_sample_profiles
    ])
    out_matrix = np.array([
        p.profile if hasattr(p, 'profile') else p for p in out_sample_profiles
    ])
    num_in_profiles, num_minutes = in_matrix.shape
    num_out_profiles = out_matrix.shape[0]

    return in_matrix, out_matrix, num_in_profiles, num_minutes, num_out_profiles

def solve_step3(epsilon, num_in_profiles, num_minutes, in_matrix,
                out_matrix, reserve_bids, expected_shortfalls, M=10000,
                consumption_lb=0, consumption_ub=600,):
    
    # Calculate maximum allowed violations for this epsilon
    # Higher epsilon allows more violations (less strict reliability)
    max_violations = int(epsilon * num_in_profiles * num_minutes)

    # Create MILP model (ALSO-X approach)
    # This is an integer programming model to maximize reserve bid
    # while limiting the number of constraint violations
    model = Model("ALSOX_tradeoff")
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output messages

    # Decision variables:
    # c: Reserve capacity bid (kW) we commit to provide
    # y: Binary variables that indicate constraint violations (1 = violation)
    c = model.addVar(lb=consumption_lb, ub=consumption_ub, name="reserve_bid")  # Bounded by max consumption
    y = model.addVars(num_in_profiles, num_minutes, vtype=GRB.BINARY, name="y")
    
    # Constraints:
    # For each profile and time point, either:
    # 1. The reserve bid c is <= available flexibility (consumption), or
    # 2. We allow a violation (y[i,t] = 1) for this profile-time pair
    for i in range(num_in_profiles):
        for t in range(num_minutes):
            model.addConstr(c - in_matrix[i, t] <= M * y[i, t])

    # Limit the total number of violations allowed
    model.addConstr(
        sum(y[i, t] for i in range(num_in_profiles) for t in range(num_minutes)) <= max_violations
    )

    # Objective: maximize the reserve bid
    # This finds the highest reserve we can offer given the reliability constraint
    model.setObjective(c, GRB.MAXIMIZE)
    model.optimize()

    # Store and evaluate results
    reserve = c.X  # Optimal reserve bid
    reserve_bids.append(reserve)

    # Calculate expected shortfall on out-of-sample data
    # Shortfall occurs when reserve bid > actual available consumption
    shortfall_matrix = np.maximum(0, reserve - out_matrix)  # Take positive values only
    expected_shortfall = np.mean(shortfall_matrix)  # Average shortfall across all samples
    expected_shortfalls.append(expected_shortfall)

    # Print progress update
    print(
        f"Epsilon: {epsilon:.2f} | P={1 - epsilon:.2f} → "
        f"Reserve: {reserve:.2f} kW, Expected shortfall: {expected_shortfall:.2f} kW"
    )

    return reserve, expected_shortfall, reserve_bids, expected_shortfalls

def plot_tradeoff(reserve_bids, expected_shortfalls, p_requirements):
    # Plot trade-off between reserve bid and expected shortfall
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()  # Create second y-axis for dual-axis plot

    # Plot reserve bids on primary y-axis (left)
    ax1.plot(p_requirements, reserve_bids, color='navy', linewidth=2.5, label='Reserve Bid')
    ax1.fill_between(p_requirements, reserve_bids, color='skyblue', alpha=0.5)  # Add shading
    ax1.set_ylabel("Reserve Bid (kW)", fontsize=14, color='navy')
    ax1.tick_params(axis='y', labelcolor='navy')
    ax1.set_ylim(210, max(reserve_bids) + 10)  # Set y-axis limits for reserve bids

    # Plot expected shortfall on secondary y-axis (right)
    ax2.plot(p_requirements, expected_shortfalls, color='darkorange', linewidth=2.5, label='Expected Shortfall')
    ax2.fill_between(p_requirements, expected_shortfalls, color='moccasin', alpha=0.5)  # Add shading
    ax2.set_ylabel("Expected Shortfall (kW/min)", fontsize=14, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, max(expected_shortfalls) * 1.1)  # Set y-axis limits with some margin

    # X-axis shows reliability requirements (decreasing from left to right)
    ax1.set_xlabel("P Requirement", fontsize=14)
    ax1.set_xlim(1.0, 0.8)  # X-axis from P100 to P80
    ax1.set_xticks([1.00, 0.95, 0.90, 0.85, 0.80])
    ax1.set_xticklabels(["1.00", "0.95", "0.90", "0.85", "0.80"], fontsize=12)
    ax1.tick_params(axis='x', labelsize=12)

    # Add title and grid
    ax1.set_title("Trade-off Between Reserve Bid and Expected Shortfall", fontsize=16)
    ax1.grid(True, axis='y')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

    plt.tight_layout()  # Adjust layout to prevent overlap
    save_path = 'part2/results/step3/figures/trade_off.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Display the plot