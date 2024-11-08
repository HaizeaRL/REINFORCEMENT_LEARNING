'''
DEEP Q-LEARNING AGENT - ENVIRONMENT EXECUTION FUNCTIONS 
'''
import os
import sys
from copy import deepcopy

# Add the parent directory (where modules is located) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import deep_q_learner as dql
from modules import environment as env


def agent_deep_q_learning(learner=dql.DeepQLearner,  environment = env.Environment,
              num_episodes=10, discount_factor=0.1, explotation_ratio=0.95, verbose=False):
    """
    Method that execute agents deep q-learning process.

    Parameters:       
        learner (Learner): The corresponding a reinforcement learning algorithm for the agent.
        environment (Environment): Environment to learn the agent.
        num_episodes (int): Number of times the agent is executed (or learns) in the environment.
        discount_factor (float): Discount factor to weigh future rewards. 
            (0 = short-term focus, 1 = long-term focus).
        explotation_ratio (float): Explotation ratio, controlling the trade-off between exploration and exploitation.
        verbose (Boolean): flag that refers whether debug information must be printed or not.

    Returns:
        None: Prints execution results. 
    """  
    # get initial point to use for the environment reset & path correct visualization
    src = environment.state

    # Deep Q-Learning algorithm instance
    learner = learner(environment=environment,
                      max_memory=100, # DETERMINE BEST VALUE
                      discount_factor=discount_factor,
                      explotation_rate=explotation_ratio) # DETERMINE BEST VALUE

    # Episode variable to print learning process result
    last_episode = None

    # Iterate for episodes
    for n_episode in range(0, num_episodes):
        print(f"Iterating episode: {n_episode +1}")
        # initialize variables
        state = environment.reset(src)
        is_final_state = False
        num_steps_episode = 0

        # start learning process
        while not is_final_state:
            old_state = state[:]
            
            # Select action, according to explotation ratio.
            next_action = learner.get_next_action(state=old_state)       

            # Execute the action.
            new_state, reward, is_final_state = environment.step(next_action) 

            # Execute learning process, for each step saves experience replay data
            # Select action batch and learns how to adjust weights
            learner.update(environment=environment,  
                           state=deepcopy(old_state),
                           action=next_action,
                           reward=reward,
                           new_state=deepcopy(new_state),
                           is_final_state=is_final_state,
                           num_episode=n_episode + 1)
            
             # Episode step sum
            num_steps_episode += 1 

        # Save last information
        last_episode = {'episode': environment,
                        'learner': learner}

        if verbose:
            # Print episode steps and reward
            print('EPISODE {} - Actions: {} - Reward: {}'
                  .format(n_episode + 1, num_steps_episode, environment.total_reward))

    print_process_info(last_episode=last_episode, start_position = src)


def print_process_info(last_episode,  start_point, print_q_table=True, 
                       print_best_values_states=True, print_best_actions_states=True,
                       print_steps=True, print_path=True):
    """
    Print execution information.

    Parameters:       
        last_episode (Learner): Last episode action steps.
        start_point (List): Agent start point to set correctly to path visualization.
        print_q_table (Boolean): Prints q-table. Default = True.
        print_best_values_states (Boolean): Prints best Q-values of each state. Default = True.
        print_best_actions_states (Boolean): Prints best steps of each state. Default = True.
        print_steps (Boolean): Prints steps that follows in the best episode. Default = True.
        print_path (Boolean): Prints followed path. Default = True.
        
    Returns:
        None: Prints deep q-learning reinforcement learning process results.
    """
    if print_q_table:
        print('\nQ_TABLE:')
        last_episode['learner'].print_q_table()

    if print_best_values_states:
        print('\nBEST Q_TABLE VALUES:')
        last_episode['learner'].print_best_values_states()

    if print_best_actions_states:
        print('\nBEST ACTIONS:')
        last_episode['learner'].print_best_actions_states()

    if print_steps:
        print('\nSTEPS: \n   {}'.format(last_episode['episode'].actions_done))

    if print_path:
        print('\nPATH:')
        last_episode['episode'].print_path_episode(start_point)