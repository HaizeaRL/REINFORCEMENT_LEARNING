import os
import sys


# Add the parent directory (where modules is located) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import environment_creation_functions as ecf
from modules import environment as env
from modules import learners as learn
from modules import environment_agent_execution as eae




# create environment
print(f"CREATING ENVIRONMENT MAZE FOR THE AGENT")
dim, grid_matrix_positions, barriers, src, dest, reward_matrix  = ecf.create_environment(False)
environment = env.Environment(reward_matrix, list(src), list(dest)) # convert states from tuple into list

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

'''# visualize created environment
print(f"ENVIRONMENT VISUALIZATION:")
ecf.visualize_environment(dim, barriers, src, dest)'''

# SHORT-TERM FOCUS USING QLEARNER
episodes_list, best_episode = eae.agent_learning(learner=learn.QLearner, # Learning algorithm
                                        environment= environment, # Recently created environment
                                        num_episodes=30,
                                        learning_rate=0.1,
                                        discount_factor=0.1,    # Near 0, learn to move to the next most rewarding state
                                        ratio_exploration=0.05, # Greedy control: Explore 5% and explote %95
                                        verbose=True)

eae.print_process_info(best_episode=best_episode)