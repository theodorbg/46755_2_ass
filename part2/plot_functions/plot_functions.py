import matplotlib.pyplot as plt
import numpy as np
import os

def plot_consumption_profiles(in_sample_profiles, out_sample_profiles):
    
    for i in range(10):
        plt.plot(np.arange(0, 60, 1), in_sample_profiles[i].profile[:60], label=f'Profile {i+1}')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Consumption (kW)')
    plt.title('In-sample Consumption Profiles')
    plt.legend()
    plt.grid()
    # Save to figures one directory up from the current file
    
    save_path = 'part2/results/step1/figures/in_sample_profiles.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved in-sample profiles to {save_path}")
    plt.close()
    

    # Visualize the first 10 out-of-sample profiles
    for i in range(10):
        plt.plot(np.arange(0, 60, 1), out_sample_profiles[i].profile[:60], label=f'Profile {i+1}')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Consumption (kW)')
    plt.title('Out-of-sample Consumption Profiles')
    plt.legend()
    plt.grid()
    save_path = 'part2/results/step1/figures/out_sample_profiles.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved out-of-sample profiles to {save_path}")
    plt.close()