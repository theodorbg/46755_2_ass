import random
from gurobipy import Model, GRB
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# Generate 300 random consumption profiles
# Each profile is:
#   - 1 hour
#   - Resolution: 1 minute
#   - lower bound: 220kW
#   - upper bound: 600kW
#   - change in consumption between two minutes <= 35kw
#   - |dC/dt| <= 35kw


random.seed(39)  # Set a seed for reproducibility
class ConsumptionProfile:
    def __init__(self, lower_bound: float, upper_bound: float, max_change: float, resolution: int, duration: int):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_change = max_change
        self.resolution = resolution
        self.duration = duration
        self.profile = self.generate_profile()

    def generate_profile(self):
        

        profile = [random.uniform(self.lower_bound, self.upper_bound)]
        for _ in range(1, self.duration * 60 // self.resolution):
            change = random.uniform(-self.max_change, self.max_change)
            new_value = profile[-1] + change
            if new_value < self.lower_bound:
                new_value = self.lower_bound
            elif new_value > self.upper_bound:
                new_value = self.upper_bound
            profile.append(new_value)
        return profile


# Generate 300 random consumption profiles
consumption_profiles = [ConsumptionProfile(220, 600, 35, 1, 1) for _ in range(300)]

# Pick out 100 random profiles for in-sample and 200 for out-of-sample
in_sample_profiles = random.sample(consumption_profiles, 100)
out_sample_profiles = [profile for profile in consumption_profiles if profile not in in_sample_profiles]

# Verify the number of profiles
# print(f"Number of in-sample profiles: {len(in_sample_profiles)}")
# print(f"Number of out-of-sample profiles: {len(out_sample_profiles)}")

# # Verify the profile lengths
# for i, profile in enumerate(in_sample_profiles):
#     print(f"In-sample profile {i+1} length: {len(profile.profile)}")
# for i, profile in enumerate(out_sample_profiles):
#     print(f"Out-of-sample profile {i+1} length: {len(profile.profile)}")

# # Verify the profile bounds
# for i, profile in enumerate(in_sample_profiles):
#     print(f"In-sample profile {i+1} bounds: {min(profile.profile)} - {max(profile.profile)}")
# for i, profile in enumerate(out_sample_profiles):
#     print(f"Out-of-sample profile {i+1} bounds: {min(profile.profile)} - {max(profile.profile)}")

# Verify the profile changes are within the limits
for i, profile in enumerate(in_sample_profiles):
    changes = [abs(profile.profile[j] - profile.profile[j-1]) for j in range(1, len(profile.profile))]
    if any(change > profile.max_change for change in changes):
        print(f"In-sample profile {i+1} has changes exceeding the limit.")

# Visualize the first 10 in-sample profiles

for i in range(10):
    plt.plot(np.arange(0, 60, 1), in_sample_profiles[i].profile[:60], label=f'Profile {i+1}')
plt.xlabel('Time (minutes)')
plt.ylabel('Consumption (kW)')
plt.title('In-sample Consumption Profiles')
plt.legend()
plt.grid()
plt.show()

# Visualize the first 10 out-of-sample profiles
for i in range(10):
    plt.plot(np.arange(0, 60, 1), out_sample_profiles[i].profile[:60], label=f'Profile {i+1}')
plt.xlabel('Time (minutes)')
plt.ylabel('Consumption (kW)')
plt.title('Out-of-sample Consumption Profiles')
plt.legend()
plt.grid()
plt.show()


print('#################################')

# %% CVaR
print("\n=== CVaR===")

from gurobipy import Model, GRB
import numpy as np

# Parameters
epsilon = 0.1
profiles_matrix = np.array([p.profile for p in in_sample_profiles])  # shape: (num_profiles, 60)
num_profiles, num_minutes = profiles_matrix.shape
N = num_profiles * num_minutes

# Model
model = Model("CVaR_P90_SlideFormulation")
model.setParam('OutputFlag', 0)

# Decision variables
c = model.addVar(lb=0, ub=600, name="reserve_bid")     # c↑
beta = model.addVar(ub=0, name="beta")                 # β ≤ 0
zeta = model.addVars(num_profiles, num_minutes, lb=0, name="zeta")  # ζ_m,ω ≥ 0

# Objective: maximize reserve bid
model.setObjective(c, GRB.MAXIMIZE)

# Constraint: shortfall definition
for i in range(num_profiles):
    for t in range(num_minutes):
        model.addConstr(c - profiles_matrix[i, t] <= zeta[i, t], f"shortfall_{i}_{t}")
        model.addConstr(beta <= zeta[i, t], f"beta_bound_{i}_{t}")

# Constraint: average zeta <= (1 - ε) * β
model.addConstr(
    (1 / N) * sum(zeta[i, t] for i in range(num_profiles) for t in range(num_minutes)) <= (1 - epsilon) * beta,
    "cvar_constraint"
)

# Solve
model.optimize()

# Output
r_cvar_dtu = c.X
print("=== CVaR (DTU Lecture Formulation) ===")
print(f"Optimal reserve capacity bid (CVaR) under P90: {r_cvar_dtu:.2f} kW")



print('#################################')

# %% ALSO - X
print("\n=== ALSO - X ===")

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

# Output
r_alsox_binary = c.X
print("=== ALSO-X (MILP with binary vars) ===")
print(f"Optimal reserve capacity bid (ALSO-X MILP) under P90: {r_alsox_binary:.2f} kW")


import numpy as np
import matplotlib.pyplot as plt

# Step 1: Extract matrix
profiles_matrix = np.array([p.profile if hasattr(p, 'profile') else p for p in in_sample_profiles])
num_profiles, num_minutes = profiles_matrix.shape
upward_flex = profiles_matrix.copy()

# Step 2: Probability matrix setup
power_axis = np.linspace(0, 600, 200)
prob_matrix = np.zeros((len(power_axis), num_minutes))

for t in range(num_minutes):
    for i, p in enumerate(power_axis):
        prob_matrix[i, t] = np.mean(upward_flex[:, t] >= p)

# Step 3: Reserve bids
r_cvar = r_cvar_dtu
r_alsox = r_alsox_binary

# Step 4: P90 line
p90_line = np.percentile(upward_flex, 10, axis=0)  # 10th percentile = P90 reserve limit

# Step 5: Plot
plt.figure(figsize=(12, 5))
extent = [0, 60, power_axis[0], power_axis[-1]]
plt.imshow(prob_matrix[::-1, :], aspect='auto', extent=extent, cmap='inferno', vmin=0, vmax=1)
plt.colorbar(label='Probability')

# Plot overlays
plt.plot(np.arange(num_minutes), upward_flex.mean(axis=0), color='lime', label='Mean', linewidth=2.5)
plt.plot(np.arange(num_minutes), p90_line, color='white', linestyle='--', linewidth=2.5, label='P90 Level')
plt.hlines(r_cvar, 0, 60, colors='cyan', linestyles='--', linewidth=2.5, label=f'CVaR Reserve ({r_cvar:.0f} kW)')
plt.hlines(r_alsox, 0, 60, colors='deepskyblue', linestyles='-.', linewidth=2.5,  label=f'ALSO-X Reserve ({r_alsox:.0f} kW)')

# Final touches
plt.title("Upwards Flexibility $F^\\uparrow$ with P90 and Reserve Bids", fontsize=16)
plt.xlabel("Time of Hour [min]", fontsize=14)
plt.ylabel("Power [kW]", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

print('#################################')

# %% Verification of the P90 Requirement Using Out-of-Sample Analysis:


print("\n===Verification of the P90 Requirement Using Out-of-Sample Analysis ===")

print('\n#################################')
print("=== Step 2.2: P90 Verification for In-sample and Out-of-Sample ===")

# Extract reserve bids
reserve_cvar = r_cvar_dtu
reserve_alsox = r_alsox_binary

# Create helper function
def verify_p90(profiles, reserve_bid, label):
    matrix = np.array([p.profile if hasattr(p, 'profile') else p for p in profiles])
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

# CVaR
verify_p90(in_sample_profiles, reserve_cvar, "CVaR (In-sample)")
verify_p90(out_sample_profiles, reserve_cvar, "CVaR (Out-of-sample)")

# ALSO-X
verify_p90(in_sample_profiles, reserve_alsox, "ALSO-X (In-sample)")
verify_p90(out_sample_profiles, reserve_alsox, "ALSO-X (Out-of-sample)")

def compute_shortfalls(profiles, reserve_bid, label):
    matrix = np.array([p.profile if hasattr(p, 'profile') else p for p in profiles])
    shortfalls = np.maximum(0, reserve_bid - matrix)

    total_shortfall_kW = np.sum(shortfalls)
    avg_shortfall_per_minute = np.mean(shortfalls[shortfalls > 0]) if np.any(shortfalls > 0) else 0.0
    minutes_with_shortfall = np.sum(shortfalls > 0)

    print(f"\n=== Shortfall Analysis: {label} ===")
    print(f"Reserve bid: {reserve_bid:.2f} kW")
    print(f"Total minutes with shortfall: {minutes_with_shortfall}")
    print(f"Total shortfall (kW): {total_shortfall_kW:.2f}")
    print(f"Average shortfall (only when > 0): {avg_shortfall_per_minute:.2f} kW")

    return shortfalls  # optionally return for further use

# Compute shortfalls for in-sample
shortfalls_cvar_in = compute_shortfalls(in_sample_profiles, r_cvar_dtu, "CVaR (In-sample)")
shortfalls_alsox_in = compute_shortfalls(in_sample_profiles, r_alsox_binary, "ALSO-X (In-sample)")

# Compute shortfalls for out-of-sample
shortfalls_cvar_out = compute_shortfalls(out_sample_profiles, r_cvar_dtu, "CVaR (Out-of-sample)")
shortfalls_alsox_out = compute_shortfalls(out_sample_profiles, r_alsox_binary, "ALSO-X (Out-of-sample)")

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

# Plot with filled area
plt.figure(figsize=(8, 5))
plt.plot(p_requirements, reserves, color='navy', linewidth=2.5, label='Reserve Bid (line)')
plt.fill_between(p_requirements, reserves, color='skyblue', alpha=0.5, label='Reserve Area')

# Styling
plt.title("Reserve Bid vs P Requirement",fontsize=16)
plt.xlabel("P Requirement",fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Optimal Reserve Bid (kW)", fontsize=14)
plt.ylim(210, max(reserves) + 10)
plt.gca().invert_xaxis()
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.show()


# === Plot Expected Shortfall ===
plt.figure(figsize=(8, 5))
plt.plot(p_requirements, expected_shortfalls, color='darkorange', linewidth=2.5, label='Expected Shortfall')
plt.fill_between(p_requirements, expected_shortfalls, color='moccasin', alpha=0.5, label='Shortfall Area')
plt.title("Expected Shortfall vs P Requirement",fontsize=16)
plt.xlabel("P Requirement", fontsize=14)
plt.ylabel("Expected Shortfall (kW per minute)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, max(expected_shortfalls) * 1.1)
plt.gca().invert_xaxis()
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.show()

#Combined
import matplotlib.pyplot as plt

# Set up figure and twin axes
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

# Plot Reserve Bid (left y-axis)
ax1.plot(p_requirements, reserve_bids, color='navy', linewidth=2.5, label='Reserve Bid')
ax1.fill_between(p_requirements, reserve_bids, color='skyblue', alpha=0.5)
ax1.set_ylabel("Reserve Bid (kW)", fontsize=14, color='navy')
ax1.tick_params(axis='y', labelcolor='navy')
ax1.set_ylim(210, max(reserve_bids) + 10)

# Plot Expected Shortfall (right y-axis)
ax2.plot(p_requirements, expected_shortfalls, color='darkorange', linewidth=2.5, label='Expected Shortfall')
ax2.fill_between(p_requirements, expected_shortfalls, color='moccasin', alpha=0.5)
ax2.set_ylabel("Expected Shortfall (kW/min)", fontsize=14, color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')
ax2.set_ylim(0, max(expected_shortfalls) * 1.1)

# X-axis: P Requirement
ax1.set_xlabel("P Requirement", fontsize=14)
ax1.set_xlim(min(p_requirements), max(p_requirements))
ax1.invert_xaxis()
clean_ticks = np.round(np.linspace(1.0, 0.8, 10), 2)  # [1.0, 0.95, 0.9, 0.85, 0.8]
ax1.set_xticks(clean_ticks)
ax1.set_xticklabels([f"{t:.2f}" for t in clean_ticks])
ax1.tick_params(axis='x', labelsize=12)

# Grid and title
ax1.set_title("Trade-off Between Reserve Bid and Expected Shortfall", fontsize=16)
ax1.grid(True, axis='y')

# Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=11)

# Final layout
plt.tight_layout()
plt.show()
