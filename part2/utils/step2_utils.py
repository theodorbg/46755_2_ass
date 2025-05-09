import random
from gurobipy import Model, GRB
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import os

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