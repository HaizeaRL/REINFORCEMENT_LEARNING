'''
AGENT LEARNING FUNCTIONS:

Apply corresponding learning function in agent's environment.
Q-LEARNING and SARSA algorithms.
'''

import numpy as np
import pandas as pd
from copy import deepcopy

class Learner(object):

    def __init__(self, environment, learning_rate=0.1, discount_factor=0.1, ratio_exploration=0.05):
        """
        Class implementing a reinforcement learning algorithm. Abstract function update is updated by 
        selected algorithm.

        Parameters:
            environment (Environment): The environment in which the algorithm will interact and take actions.
            learning_rate (float): Learning rate that determines the extent of learning in each step.
            discount_factor (float): Discount factor to weigh future rewards. 
                (0 = short-term focus, 1 = long-term focus).
            ratio_exploration (float): Exploration ratio, controlling the trade-off between exploration and exploitation.
        """
        self.environment = environment
        # Q-table
        self.q_table = [[
                          [0.0 for _ in environment.actions]  # Initialize a Q-value for each action
                          for _ in range(len(environment.rewards))  # For each row (state along one axis)
                        ]
                        for _ in range(len(environment.rewards[0]))  # For each column (state along the other axis)
                       ]
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.ratio_exploration = ratio_exploration

    @property
    def name(self):
        return 'default'

    def get_next_action(self, state):
        """
        Method that selects the next action to take:
            Random (exploration) -> if the exploration ratio is below the threshold.
            Best Action (explotation) -> if the exploration ratio is higher than the threshold.

        Parameters:
            state: Agent state

        Returns:
            next_action: Agent next action.
        """
        if np.random.uniform() < self.ratio_exploration:
            # selection of possible actions at random.
            next_action = np.random.choice(list(self.environment.actions))
        else:  
            # find the maximum Q-value for all actions
            max_q_value = np.array(self.q_table[state[0]][state[1]]).max()

            # Get the indices of actions that have the highest Q-value at the current state.
            best_action_index = np.flatnonzero(self.q_table[state[0]][state[1]] == max_q_value)
            
            # selects one of these indices at random
            idx_action = np.random.choice(best_action_index)

            # select next action according to selected index
            next_action = list(self.environment.actions)[idx_action]

        return next_action
    
    def update(self, **kwargs):
        """
        Abstract function
        Parameters:
            distinct parameters depending of the algorithm.
        """
        pass
        
    def print_q_table(self):
        """
        Function that prints q-table or policy.
        """
        q_table = []
        for state_x, actions in enumerate(self.q_table):
            for state_y, action in enumerate(actions):
                # deepcopy actions values
                q = deepcopy(action)
                # add state to action in first place. Example: x0,y0 actions_values.
                q.insert(0, 'x{},y{}'.format(state_x, state_y))
                # add state to global table
                q_table.append(q)

        # print resulted table as dataframe        
        print(pd.DataFrame(data=q_table,
                           columns=['State', 'Up', 'Down', 'Left', 'Right'])
              .to_string(index=False))
        
    def print_best_values_states(self):
        """
        Function that prints best value of the best option to be performed in each of the states.
        """
        best = [[max(vi) for vi in row] for row in self.q_table] # obtain max values of each state 

        # print result as dataframe
        print(pd.DataFrame(data=np.array([np.array(xi) for xi in best]), # positionate best values in each state 
                           index=["x{}".format(str(i)) for i in range(len(best))], # add state row values
                           columns=["y{}".format(str(i)) for i in range(len(best[0]))])) # add state col values
        
    def print_best_actions_states(self):
        """
        Function that prints best option to take in each action.
        """
        # obtain best actions of each state acording to max value
        best = [[list(self.environment.actions)[np.argmax(col)] for col in row] for row in self.q_table]
        print(pd.DataFrame(data=np.array([np.array(xi) for xi in best]),  # positionate best action in each state 
                           index=["x{}".format(str(i)) for i in range(len(best))],  # add state row values
                           columns=["y{}".format(str(i)) for i in range(len(best[0]))])) # add state col 
        
class QLearner (Learner):

    @property
    def name(self):
        return 'QLearner'
    
    def update (self, environment, old_state, action_taken,
                reward_action_taken,  new_state, is_final_state, **kwargs):
        """
        Function that applies the Q-Learning algorithm as a strategy.

        Algorithm:
            1. For each non-terminal state, select the next action according to an exploration-exploitation strategy
              (greedy control).
            2. Update the state and receive the associated reward.
            3. Using the reward obtained calculate the updated Q-value.
            4. Adjust the Q-table accordingly based on the update rule specific to SARSA.

        Parameters:
            environment (Environment): The environment in which the algorithm will interact and take actions.
            old_state: The agent's current state.
            action_taken: The action chosen by the agent.
            reward_action_taken: The reward received for taking the specified action.
            new_state: The new state reached by the agent after taking the action.
            is_final_state: Boolean indicating whether the agent has reached the terminal state.

        Returns:
            None: Updates the Q-table in place according to the Q-Learning algorithm.
        """
        # Obtain index of taken action.
        idx_action_taken = list(environment.actions).index(action_taken)

        # Get the Q-value of the action that was taken
        actual_q_value_options = self.q_table[old_state[0]][old_state[1]]
        actual_q_value = actual_q_value_options[idx_action_taken]

        # Retrieve the Q-values for the new state after taking the action
        future_q_value_options = self.q_table[new_state[0]][new_state[1]]
        
        # Calculate the expected future Q-value using the reward received and the maximum future Q-value
        future_max_q_value = reward_action_taken + self.discount_factor * max(future_q_value_options)       
        if is_final_state:
            future_max_q_value = reward_action_taken    # Maximum reward if reach to final state

        # Update the Q-table by applying the Q-Learning update rule. 
        # The new value for the action taken is based on the temporal difference
        self.q_table[old_state[0]][old_state[1]][idx_action_taken] = \
            actual_q_value + self.learning_rate * (future_max_q_value - actual_q_value)        

class SARSALearner (Learner):

    @property
    def name(self):
        return 'SARSALearner'

    def update(self, environment, old_state, action_taken,
                reward_action_taken,  new_state, new_action, is_final_state, **kwargs):
        """
        Function that applies the SARSA algorithm as a strategy.

        Algorithm:
            1. For each non-terminal state, select the next action according to an exploration-exploitation strategy
              (greedy control).
            2. Update the state and receive the associated reward.
            3. From this new state, select the next action according to the same exploration-exploitation strategy.
            4. Using the reward obtained and the Q-value of the next action in the next state, calculate the updated 
            Q-value.
            5. Adjust the Q-table accordingly based on the update rule specific to SARSA.
        Selects the action that would be taken after we move to the next state.

        Parameters:
            environment (Environment): The environment in which the algorithm will interact and take actions.
            old_state: The agent's current state.
            action_taken: The action chosen by the agent.
            reward_action_taken: The reward received for taking the specified action.
            new_state: The new state reached by the agent after taking the action.
            new_action: The new action to be taken after move to the new state.
            is_final_state: Boolean indicating whether the agent has reached the terminal state.

        Returns:
            None: Updates the Q-table in place according to the SARSA algorithm.
        """
        # Obtain index of taken action.
        idx_action_taken = list(environment.actions).index(action_taken)

        # Get the Q-value of the action that was taken
        actual_q_value_options = self.q_table[old_state[0]][old_state[1]]
        actual_q_value = actual_q_value_options[idx_action_taken]

        # Retrieve the Q-values for the new state after taking the action
        future_q_value_options = self.q_table[new_state[0]][new_state[1]]

        # Calculate the expected future Q-value using the reward received and the Q-value of the new action taken
        idx_new_action_taken = list(environment.actions).index(new_action)
        future_new_action_q_value = \
            reward_action_taken + self.discount_factor * future_q_value_options[idx_new_action_taken]
        if is_final_state:           
            future_new_action_q_value = reward_action_taken # Maximum reward if reach to final state

        # Update the Q-table by applying the SARSA update rule. 
        # The new Q-value for the action taken is based on the next action and the temporal difference
        self.q_table[old_state[0]][old_state[1]][idx_action_taken] = \
            actual_q_value + self.learning_rate * (future_new_action_q_value - actual_q_value)