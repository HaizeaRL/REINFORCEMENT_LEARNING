import os
import sys

# Add the parent directory (where modules is located) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import environment_creation_functions as ecf
from modules import environment as env




# create environment
print(f"CREATING ENVIRONMENT MAZE FOR THE AGENT")
dim, grid_matrix_positions, barriers, src, dest, reward_matrix  = ecf.create_environment(False)
environment = env.Environment(reward_matrix, src, dest)

# print initial values
print(f"\nENVIRONMENT PARAMETERS:")
print(f"Possible actions: {environment.actions}")
print(f"Reward matrix:")
ecf.visualize_matrix (dim, environment.rewards)
print(f"Penalty value: {environment.action_penalty}")
print(f"Initial state: {environment.state}")
print(f"Final state: {environment.final_state}")
print(f"Initial reward score: {environment.total_reward}")
print(f"Actions: {environment.actions_done}")

# visualize created environment
print(f"ENVIRONMENT VISUALIZATION:")
ecf.visualize_environment(dim, barriers, src, dest)
