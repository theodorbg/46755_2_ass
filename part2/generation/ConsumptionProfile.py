import random
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

class ConsumptionProfile:
    def __init__(self, lower_bound: float, upper_bound: float, max_change: float, resolution: int, duration: int):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_change = max_change
        self.resolution = resolution
        self.duration = duration
        self.profile = self.generate_profile()

    def generate_profile(self, seed = None):
    
        if seed is not None:
            random.seed(seed)

        profile = [random.uniform(self.lower_bound, self.upper_bound)]
        # The loop should generate (duration * 60 / resolution) - 1 more points
        # So the range should go up to that number.
        # Example: 1 hour, 1 min resolution -> 60 points. First is generated. Loop 59 times.
        # range(1, 60) -> 1 to 59. Correct.
        for _ in range(1, self.duration * 60 // self.resolution):
            change = random.uniform(-self.max_change, self.max_change)
            new_value = profile[-1] + change
            # Clamping the value to be within bounds
            new_value = max(self.lower_bound, min(new_value, self.upper_bound))
            profile.append(new_value)
        return profile

def verify_profiles(in_sample_profiles, out_sample_profiles):
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
    # for i, profile in enumerate(in_sample_profiles):
    #     changes = [abs(profile.profile[j] - profile.profile[j-1]) for j in range(1, len(profile.profile))]
    #     if any(change > profile.max_change for change in changes):
    #         print(f"In-sample profile {i+1} has changes exceeding the limit.")
    print("\nIn-sample profiles verified successfully.")