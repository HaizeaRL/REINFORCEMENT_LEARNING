import numpy as np
import pandas as pd
from copy import deepcopy

class Environment(object):
    def __init__(self, reward_matrix, src, dest, action_penalty=-1.0):
        """
        Class that represents and controls the environment.
        Parameters:
            reward_matrix (matrix): Reward matrix to apply Reinforcement Learning in it.           
            src (tuple): Start position in the maze.
            dest (tuple): End position in the maze.
            step_penalty (int): Reward discount factor for each action.   
        """
        self.actions = {'Up': [-1, 0],
                        'Down': [1, 0],
                        'Left': [0, -1],
                        'Right': [0, 1]}
        self.rewards = reward_matrix                 # Environment reward matrix
        self.action_penalty = action_penalty         # Penalty for each step taken
        self.state = src                             # Agent current state.     
        self.final_state = dest                      # Agents destination. When the agent arrives, the episode ends.
        self.total_reward = 0.0                      # Episode reward counter
        self.actions_done = []                       # List of steps (actions) performed in each episode. 

    def reset(self, src):
        """
        Method that resets the environment variables and returns the initial state.

        Returns:    
            initial position of the maze.
        """
        self.total_reward = 0.0    # Initialize episode reward counter
        self.state = list(src)   # Positionate the agent in the initial state
        self.actions_done = []     # Initialize the list of steps (actions)

        return self.state
    
    def stay_within_bounds(self):
        """
        Method that controls whether the agent is withing bounds.

        Returns:    
            Correspondin state.
        """
        # Limit vertical positions according to grid limit (up and down)
        self.state[0] = max(0, min(self.state[0], len(self.rewards) - 1))
        # Limit horizontal positions according to grid limit (left and right)
        self.state[1] = max(0, min(self.state[1], len(self.rewards[0]) - 1))

        return  self.state[0], self.state[1]
        
    
    def __apply_action(self, action):
        """
        Method that calculates the new state from the action to be executed.

        Parameters:
            action: The action to be taken, specifying the direction of movement.

        Returns:
            tuple: The agent's current state after applying the action and adjusting for boundaries.
        """
        self.state[0] += self.actions[action][0]
        self.state[1] += self.actions[action][1]

        # Ensure the state remains within grid boundaries
        self.state[0], self.state[1] = self.stay_within_bounds()
        
        return self.state[0], self.state[1]

    def step(self, action, verbose = False):
        """
        Method that executes a given action from the set of actions {Up, Down, Left, Right}
        Guide the agent in the environment.

        Parameters:
            action: Action to be taken.

        Returns: 
            self.state (tuple): The current state
            reward (int): The obtained reward
            is_final_state (boolean): Whether we have reached the final state.
            verbose (boolean): Whether debug logs must be showed or not.
        """
        old_state = deepcopy(self.state)  # save current state          
        self.__apply_action(action)       # Perform the action (change of state)           

        # check if still in same state. Cases: new state after action is not in bound or is passed state.
        reward = 0
        if not (np.array_equal(self.state, old_state)):

            # Add action to action list
            self.actions_done.append(self.state[:])  

            # Calculate the reward for the action taken
            reward = self.rewards[self.state[0]][self.state[1]] + self.action_penalty  

            # Add the current reward to the total reward for the episode
            self.total_reward += reward     
        else:
            # penalize out of bound with -50
            reward = -50
            # Add the current reward to the total reward for the episode
            self.total_reward += reward  


        # Check if we are in the final state or destination.
        is_final_state = np.array_equal(self.state, self.final_state)              
                                  
        return self.state, reward, is_final_state       

    def print_path_episode(self, src):
        """
        Method that prints the path followed by the agent.

        Parameters:
            src: Agent initical position to set as step 0.

        Returns: 
            None: prints the path followed by the agent.
        """
      
        path = [['-' for _ in range(len(self.rewards))] for _ in range(len(self.rewards[0]))]
        path[src[0]][src[1]] = '0'

        # Loop through actions and track repeated visits
        for index, step in enumerate(self.actions_done):
            if path[step[0]][step[1]] == '-':
                path[step[0]][step[1]] = str(index + 1)
            else:
                # Append the step number if the position has been visited before
                path[step[0]][step[1]] += f',{index + 1}'

        # Convert the path to a DataFrame for better visualization
        path_df = pd.DataFrame(data=np.array(path),
                               index=[f"x{i}" for i in range(len(path))],
                               columns=[f"y{i}" for i in range(len(path[0]))])
        
        print(path_df)

