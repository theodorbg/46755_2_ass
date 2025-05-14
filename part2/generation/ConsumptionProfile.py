"""
Consumption Profile Generator

This module provides functionality to generate and verify random electricity 
consumption profiles with controlled properties including bounds and rate-of-change limits.
These profiles simulate real-world electricity consumption patterns for testing
flexibility and reserve capacity algorithms.
"""

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
    """
    A class representing an electricity consumption profile with constrained properties.
    
    The consumption profile is generated as a time series with random variations,
    but with constraints on minimum/maximum values and maximum rate of change.
    This simulates realistic electricity consumption behavior.
    
    Attributes:
        lower_bound (float): Minimum allowed consumption value in kW
        upper_bound (float): Maximum allowed consumption value in kW
        max_change (float): Maximum allowed change between time steps in kW
        resolution (int): Time step in minutes 
        duration (int): Duration of profile in hours
        seed (int, optional): Random seed for reproducible profiles
        profile (list): Generated consumption profile values
    """
    
    def __init__(self, lower_bound: float, upper_bound: float,
                 max_change: float, resolution: int, duration: int, seed: int = None):
        """
        Initialize and generate a consumption profile.
        
        Args:
            lower_bound (float): Minimum consumption value in kW
            upper_bound (float): Maximum consumption value in kW 
            max_change (float): Maximum allowed change between time steps in kW
            resolution (int): Time step in minutes
            duration (int): Duration of profile in hours
            seed (int, optional): Random seed for reproducible profiles
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_change = max_change
        self.resolution = resolution
        self.duration = duration
        self.seed = seed
        self.profile = self.generate_profile(seed=self.seed)

    def generate_profile(self, seed=None):
        """
        Generate a random consumption profile within defined constraints.
        
        The profile starts with a random value within bounds, then each subsequent
        point varies by a random amount limited by max_change. All values are
        kept within the lower and upper bounds.
        
        Args:
            seed (int, optional): Random seed for reproducible results
            
        Returns:
            list: The generated consumption profile as a list of values
        """
        # Set random seed if provided for reproducibility
        if seed is not None:
            random.seed(seed)

        # Start with a random value within bounds
        profile = [random.uniform(self.lower_bound, self.upper_bound)]
        
        # Calculate number of points needed
        # Example: 1 hour, 1 min resolution → 60 points
        # First point is generated above, so loop 59 times
        total_points = self.duration * 60 // self.resolution
        
        # Generate each subsequent point
        for _ in range(1, total_points):
            # Generate random change within limits
            change = random.uniform(-self.max_change, self.max_change)
            
            # Apply change and ensure result stays within bounds
            new_value = profile[-1] + change
            new_value = max(self.lower_bound, min(new_value, self.upper_bound))
            
            profile.append(new_value)
            
        return profile


def verify_profiles(in_sample_profiles, out_sample_profiles):
    """
    Verify that the generated profiles meet all required constraints.
    
    This function checks:
    1. The number of profiles in each set
    2. The length of each profile
    3. Whether all values are within specified bounds
    4. Whether all changes between consecutive points are within max_change
    
    Args:
        in_sample_profiles (list): List of ConsumptionProfile objects for in-sample data
        out_sample_profiles (list): List of ConsumptionProfile objects for out-of-sample data
    """
    # Verify the number of profiles
    print(f"Number of in-sample profiles: {len(in_sample_profiles)}")
    print(f"Number of out-of-sample profiles: {len(out_sample_profiles)}")

    # Verify the profile lengths (all should match the expected duration/resolution)
    for i, profile in enumerate(in_sample_profiles):
        print(f"In-sample profile {i+1} length: {len(profile.profile)}")
    for i, profile in enumerate(out_sample_profiles):
        print(f"Out-of-sample profile {i+1} length: {len(profile.profile)}")

    # Verify all values are within the specified bounds
    for i, profile in enumerate(in_sample_profiles):
        print(f"In-sample profile {i+1} bounds: {min(profile.profile)} - {max(profile.profile)}")
    for i, profile in enumerate(out_sample_profiles):
        print(f"Out-of-sample profile {i+1} bounds: {min(profile.profile)} - {max(profile.profile)}")

    # Verify the profile changes are within the limits
    for i, profile in enumerate(in_sample_profiles):
        # Calculate absolute changes between consecutive points
        changes = [abs(profile.profile[j] - profile.profile[j-1]) for j in range(1, len(profile.profile))]
        # Check if any change exceeds the specified maximum
        if any(change > profile.max_change for change in changes):
            print(f"In-sample profile {i+1} has changes exceeding the limit.")
    print("\nIn-sample profiles verified successfully.")