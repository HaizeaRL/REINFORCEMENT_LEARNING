import os
import sys
from copy import deepcopy
import numpy as np

# Add the parent directory (where modules is located) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import environment as env
from modules import learners as learn

def agent_learning(learner=learn.Learner, environment=env.Environment,
                   num_episodes=10, learning_rate=0.1, discount_factor=0.1, ratio_exploration=0.05,
                   verbose=False):
    """
    Method that execute agents learning process.

    Parameters:       
        learner (Learner): The corresponding a reinforcement learning algorithm for the agent.
        environment (Environment): The environment in which the algorithm will interact and take actions.
        num_episodes (int): Number of times the agent is executed (or learns) in the environment.
        learning_rate (float): Learning rate that determines the extent of learning in each step.
        discount_factor (float): Discount factor to weigh future rewards. 
            (0 = short-term focus, 1 = long-term focus).
        ratio_exploration (float): Exploration ratio, controlling the trade-off between exploration and exploitation.
        verbose (Boolean): flag that refers whether debug information must be printed or not.

    Returns:
        episodes_list: All learning episodes data.
        best_episode: Best learned episode data.
    """  
    # Learning algorith instance
    learner = learner(environment=environment,
                      learning_rate=learning_rate,
                      discount_factor=discount_factor,
                      ratio_exploration=ratio_exploration)
   

    # Episodes variables
    episodes_list = []
    best_reward = float('-inf')
    best_episode = None
    
  
    # Iterate episodes
    for n_episode in range(0, num_episodes):
        # start environment for each episode.
        state = environment.reset() 
        # initialize variables
        is_final_state = np.array_equal(environment.state, environment.final_state)     
        reward = None
        num_steps_episode = 0
        while not is_final_state: # While non-terminal state
            old_state = state[:]
            print(f"_agent_learning:: old_state: {state}")
            # Select action, according to exploration ratio.
            next_action = learner.get_next_action(state=state)               
            print(f"_agent_learning::next_action: {next_action}")
            # Execute the action. Obtain new state, reward and if destination is reached.
            state, reward, is_final_state = environment.step(next_action, verbose)    
            print(f"_agent_learning::current state: {state}")
            print(f"_agent_learning::current reward: {reward}")
            print(f"_agent_learning::is_final_state: {is_final_state}")
            next_post_action = (learner.get_next_action(state)               # Select new action from the new state only If learning method is SARSA.
                                if learner.name == 'SARSALearner' else None)
            print(f"_agent_learning::next_post_action: {next_post_action}")
            input()

            learner.update(environment=environment,                          # Update the environment according to the learning algorithm.
                           old_state=old_state,
                           action_taken=next_action,
                           reward_action_taken=reward,
                           new_state=state,
                           new_action=next_post_action,
                           is_final_state=is_final_state)
            num_steps_episode += 1                                           # Episode stem sum

        # Save episode information: episode number, steps and total reward
        episodes_list.append([n_episode + 1,                                 
                              num_steps_episode, 
                              environment.total_reward])

        # obtain best reward episode.
        if environment.total_reward >= best_reward:
            best_reward = environment.total_reward
            best_episode = {'num_episode': n_episode + 1,
                            'episode': deepcopy(environment),
                            'learner': deepcopy(learner)}

        if verbose:
            # Print episode steps and reward
            print('EPISODE {} - Actions: {} - Reward: {}'
                  .format(n_episode + 1, num_steps_episode, environment.total_reward))

    return episodes_list, best_episode


def print_process_info(best_episode, print_best_episode_info=True,
                       print_q_table=True, print_best_values_states=True,
                       print_best_actions_states=True, print_steps=True, print_path=True):
    """
    Print execution best episode information.
    Q-table results
    Best Q-values of each state.
    Best steps of each state.
    Steps that follows,
    """
    if print_best_episode_info:
        print('\nBEST EPISODE:\nEPISODE {}\n\tActions: {}\n\tReward: {}'
              .format(best_episode['num_episode'],
                      len(best_episode['episode'].actions_done),
                      best_episode['episode'].total_reward))

    if print_q_table:
        print('\nQ_TABLE:')
        best_episode['learner'].print_q_table()
        
    if print_best_values_states:
        print('\nBEST Q_TABLE VALUES:')
        best_episode['learner'].print_best_values_states()

    if print_best_actions_states:
        print('\nBEST ACTIONS:')
        best_episode['learner'].print_best_actions_states()

    if print_steps:
        print('\nSteps: \n   {}'.format(best_episode['episode'].actions_done))

    if print_path:
        print('\nPATH:')
        best_episode['episode'].print_path_episode()
