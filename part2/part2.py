print('#################################')
print('\nInitializing part2.py...')
import random
from gurobipy import Model, GRB
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# %% Data generation
print('\nGenerating consumption profiles...')
random.seed(39)  # Set a seed for reproducibility

class ConsumptionProfile:
    def __init__(self, lower_bound: float, upper_bound: float,
                 max_change: float, resolution: int, duration: int):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_change = max_change
        self.resolution = resolution
        self.duration = duration
        self.profile = self.generate_profile()

    def generate_profile(self):
        profile = [random.uniform(self.lower_bound, self.upper_bound)]
        num_points = self.duration * 60 // self.resolution

        for _ in range(1, num_points):
            change = random.uniform(-self.max_change, self.max_change)
            new_value = profile[-1] + change
            new_value = min(max(new_value, self.lower_bound), self.upper_bound)
            profile.append(new_value)

        return profile


# Generate 300 random consumption profiles
consumption_profiles = [
    ConsumptionProfile(220, 600, 35, 1, 1) for _ in range(300)
]

# Pick 100 random profiles for in-sample, the rest for out-of-sample
in_sample_profiles = random.sample(consumption_profiles, 100)
out_sample_profiles = [
    profile for profile in consumption_profiles if profile not in in_sample_profiles
]

print('\nDone generating consumption profiles.')
# %% CVaR
print("\n=== Computing CVaR (P90)... ===")

# Parameters
epsilon = 0.1
profiles_matrix = np.array([p.profile for p in in_sample_profiles])  # shape: (num_profiles, 60)
num_profiles, num_minutes = profiles_matrix.shape
N = num_profiles * num_minutes

# Model
model = Model("CVaR_P90_SlideFormulation")
model.setParam('OutputFlag', 0)

# Decision variables
c = model.addVar(lb=0, ub=600, name="reserve_bid")           # c↑
beta = model.addVar(ub=0, name="beta")                       # β ≤ 0
zeta = model.addVars(num_profiles, num_minutes, lb=0, name="zeta")  # ζ_m,ω ≥ 0

# Objective: maximize reserve bid
model.setObjective(c, GRB.MAXIMIZE)

# Constraints: shortfall definition and β bound
for i in range(num_profiles):
    for t in range(num_minutes):
        model.addConstr(c - profiles_matrix[i, t] <= zeta[i, t], name=f"shortfall_{i}_{t}")
        model.addConstr(beta <= zeta[i, t], name=f"beta_bound_{i}_{t}")

# CVaR constraint: average zeta ≤ (1 - ε) * β
model.addConstr(
    (1 / N) * sum(zeta[i, t] for i in range(num_profiles) for t in range(num_minutes))
    <= (1 - epsilon) * beta,
    name="cvar_constraint"
)

# Solve
model.optimize()

# Output
r_cvar_dtu = c.X
print(f"Optimal reserve capacity bid (CVaR) under P90: {r_cvar_dtu:.2f} kW")
print("=== CVaR (P90) Computed ===")
# %% ALSO - X
print("\n=== Computing ALSO - X... ===")

# Parameters
epsilon = 0.1  # P90 constraint → 10% violation allowance
profiles_matrix = np.array([p.profile for p in in_sample_profiles])  # shape: (100, 60)
num_profiles, num_minutes = profiles_matrix.shape
max_violations = int(epsilon * num_profiles * num_minutes)
print(f"Max violations allowed: {max_violations} (10% of {num_profiles * num_minutes})")
M = 10000  # Big-M constant

# Create MILP model
model = Model("ALSOX_MILP")
model.setParam('OutputFlag', 0)

# Variables
c = model.addVar(lb=0, ub=600, name="reserve_bid")  # c↑
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
r_alsox_binary = c.X

# Plotting
# Extract matrix
profiles_matrix = np.array([p.profile if hasattr(p, 'profile') else p for p in in_sample_profiles])
num_profiles, num_minutes = profiles_matrix.shape
upward_flex = profiles_matrix.copy()

# Probability matrix setup
power_axis = np.linspace(0, 600, 200)
prob_matrix = np.zeros((len(power_axis), num_minutes))

for t in range(num_minutes):
    for i, p in enumerate(power_axis):
        prob_matrix[i, t] = np.mean(upward_flex[:, t] >= p)

# Reserve bids
r_cvar = r_cvar_dtu
r_alsox = r_alsox_binary

# P90 line
p90_line = np.percentile(upward_flex, 10, axis=0)  # 10th percentile = P90 reserve limit

# One combiend plot
plt.figure(figsize=(12, 5))
extent = [0, 60, power_axis[0], power_axis[-1]]
plt.imshow(prob_matrix[::-1, :], aspect='auto', extent=extent, cmap='inferno', vmin=0, vmax=1)
plt.colorbar(label='Probability')
plt.plot(np.arange(num_minutes), upward_flex.mean(axis=0), color='lime', label='Mean', linewidth=2.5)
plt.plot(np.arange(num_minutes), p90_line, color='white', linestyle='--', linewidth=2.5, label='P90 Level')
plt.hlines(r_cvar, 0, 60, colors='cyan', linestyles='--', linewidth=2.5, label=f'CVaR Reserve ({r_cvar:.0f} kW)')
plt.hlines(r_alsox, 0, 60, colors='deepskyblue', linestyles='-.', linewidth=2.5,  label=f'ALSO-X Reserve ({r_alsox:.0f} kW)')
plt.title("Upwards Flexibility $F^\\uparrow$ with P90 and Reserve Bids", fontsize=16)
plt.xlabel("Time of Hour [min]", fontsize=14)
plt.ylabel("Power [kW]", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

print(f"Optimal reserve capacity bid (ALSO-X MILP) under P90: {r_alsox_binary:.2f} kW")
print("=== ALSO-X Computed ===")
# %% Verification of the P90 Requirement Using Out-of-Sample Analysis:

print("\n=== Computing Verification of the P90 Requirement Using Out-of-Sample Analysis... ===")

# Extract reserve bids
reserve_cvar = r_cvar_dtu
reserve_alsox = r_alsox_binary


def verify_p90(profiles, reserve_bid, label):
    """Check whether a reserve bid satisfies the P90 requirement."""
    matrix = np.array([
        p.profile if hasattr(p, 'profile') else p for p in profiles
    ])
    num_profiles, num_minutes = matrix.shape
    total_minutes = num_profiles * num_minutes
    allowed_violations = int(0.1 * total_minutes)

    violations = np.sum(matrix < reserve_bid)
    rate = violations / total_minutes
    satisfied = violations <= allowed_violations

    print(f"\n--- {label} ---")
    print(f"Reserve bid: {reserve_bid:.2f} kW")
    print(f"Total minutes: {total_minutes}")
    print(f"Allowed (10%): {allowed_violations}")
    print(f"Violations: {violations}")
    print(f"Violation rate: {rate:.2%}")
    print(f"P90 satisfied: {'YES ✅' if satisfied else 'NO ❌'}")


# Run P90 verification
verify_p90(in_sample_profiles, reserve_cvar, "CVaR (In-sample)")
verify_p90(out_sample_profiles, reserve_cvar, "CVaR (Out-of-sample)")
verify_p90(in_sample_profiles, reserve_alsox, "ALSO-X (In-sample)")
verify_p90(out_sample_profiles, reserve_alsox, "ALSO-X (Out-of-sample)")


def compute_shortfalls(profiles, reserve_bid, label):
    """Calculate and report shortfall statistics."""
    matrix = np.array([
        p.profile if hasattr(p, 'profile') else p for p in profiles
    ])
    shortfalls = np.maximum(0, reserve_bid - matrix)

    total_shortfall_kW = np.sum(shortfalls)
    avg_shortfall = (
        np.mean(shortfalls[shortfalls > 0]) if np.any(shortfalls > 0) else 0.0
    )
    minutes_with_shortfall = np.sum(shortfalls > 0)

    print(f"\n=== Shortfall Analysis: {label} ===")
    print(f"Reserve bid: {reserve_bid:.2f} kW")
    print(f"Total minutes with shortfall: {minutes_with_shortfall}")
    print(f"Average shortfall (only when > 0): {avg_shortfall:.2f} kW")

    return shortfalls

# Compute shortfalls (in-sample)
shortfalls_cvar_in = compute_shortfalls(in_sample_profiles, reserve_cvar, "CVaR (In-sample)")
shortfalls_alsox_in = compute_shortfalls(in_sample_profiles, reserve_alsox, "ALSO-X (In-sample)")

# Compute shortfalls (out-of-sample)
shortfalls_cvar_out = compute_shortfalls(out_sample_profiles, reserve_cvar, "CVaR (Out-of-sample)")
shortfalls_alsox_out = compute_shortfalls(out_sample_profiles, reserve_alsox, "ALSO-X (Out-of-sample)")

print('\n#################################')

print("=== Step 2.3 Energinet Perspective ===")

# Step 1: Define epsilon values (P100 to P80)
epsilons = np.linspace(0.0, 0.2, 21)  # from 0% to 20% violations
reserve_bids = []
expected_shortfalls = []

# Data
in_matrix = np.array([p.profile if hasattr(p, 'profile') else p for p in in_sample_profiles])
out_matrix = np.array([p.profile if hasattr(p, 'profile') else p for p in out_sample_profiles])
num_in_profiles, num_minutes = in_matrix.shape
num_out_profiles = out_matrix.shape[0]

# Vary epsilon (1 - P requirement)
epsilons = np.linspace(0.0, 0.2, 21)  # from P100 to P80
p_requirements = 1 - epsilons
reserve_bids = []
expected_shortfalls = []

# Loop over epsilons
for epsilon in epsilons:
    max_violations = int(epsilon * num_in_profiles * num_minutes)

    # Create ALSO-X MILP model
    model = Model("ALSOX_tradeoff")
    model.setParam('OutputFlag', 0)
    c = model.addVar(lb=0, ub=600, name="reserve_bid")
    y = model.addVars(num_in_profiles, num_minutes, vtype=GRB.BINARY, name="y")
    M = 10000

    # Constraints
    for i in range(num_in_profiles):
        for t in range(num_minutes):
            model.addConstr(c - in_matrix[i, t] <= M * y[i, t])
    model.addConstr(sum(y[i, t] for i in range(num_in_profiles) for t in range(num_minutes)) <= max_violations)

    # Objective: maximize reserve bid
    model.setObjective(c, GRB.MAXIMIZE)
    model.optimize()

    # Get result
    reserve = c.X
    reserve_bids.append(reserve)

    # Compute expected shortfall on out-of-sample
    shortfall_matrix = np.maximum(0, reserve - out_matrix)
    expected_shortfall = np.mean(shortfall_matrix)
    expected_shortfalls.append(expected_shortfall)

    print(f"Epsilon: {epsilon:.2f} | P={1 - epsilon:.2f} → Reserve: {reserve:.2f} kW, Expected shortfall: {expected_shortfall:.2f} kW")

# Step 3: Plot results
# Prepare data
p_requirements = 1 - epsilons  # from P100 to P80
reserves = reserve_bids


# Set up figure and twin axes
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

# Plot reserve bids (left y-axis)
ax1.plot(p_requirements, reserve_bids, color='navy', linewidth=2.5, label='Reserve Bid')
ax1.fill_between(p_requirements, reserve_bids, color='skyblue', alpha=0.5)
ax1.set_ylabel("Reserve Bid (kW)", fontsize=14, color='navy')
ax1.tick_params(axis='y', labelcolor='navy')
ax1.set_ylim(210, max(reserve_bids) + 10)

# Plot expected shortfall (right y-axis)
ax2.plot(p_requirements, expected_shortfalls, color='darkorange', linewidth=2.5, label='Expected Shortfall')
ax2.fill_between(p_requirements, expected_shortfalls, color='moccasin', alpha=0.5)
ax2.set_ylabel("Expected Shortfall (kW/min)", fontsize=14, color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')
ax2.set_ylim(0, max(expected_shortfalls) * 1.1)

# Set up x-axis
ax1.set_xlabel("P Requirement", fontsize=14)
ax1.set_xlim(1.0, 0.8)
ax1.set_xticks([1.00, 0.95, 0.90, 0.85, 0.80])
ax1.set_xticklabels(["1.00", "0.95", "0.90", "0.85", "0.80"], fontsize=12)
ax1.tick_params(axis='x', labelsize=12)

# Title and grid
ax1.set_title("Trade-off Between Reserve Bid and Expected Shortfall", fontsize=16)
ax1.grid(True, axis='y')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

# Show plot
plt.tight_layout()
plt.show()