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
        self.initial_state = src                     # Agent initial state.
        self.final_state = dest                      # Agents destination. When the agent arrives, the episode ends.
        self.total_reward = 0.0                      # Episode reward counter
        self.actions_done = []                       # List of steps (actions) performed in each episode. 

    def reset(self):
        """
        Method that resets the environment variables and returns the initial state.

        Returns:    
            initial position of the maze.
        """
        self.total_reward = 0.0    # Initialize episode reward counter
        self.state = self.initial_state    # Positionate the agent in the initial state
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
            Avoids state repetition.
        """
        # TODO: correct the logic. Para que no repita la misma accion izq - derecha seguido pero que no quede bloqueado
        # Calculate the intended new position based on the action's movement
        new_x = self.state[0] + self.actions[action][0]
        new_y = self.state[1] + self.actions[action][1]

        # Update the state only if the new position has not been visited
        if [new_x, new_y] not in self.actions_done:
            self.state[0] = new_x
            self.state[1] = new_y

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

            if verbose:
                print(f"_step::NEW REWARD: {reward}")  
                print(f"_step::TOTAL REWARD: {self.total_reward}")
            
        else:
            print("Action not taken. Step is out bound or repeated.") 

        # Check if we are in the final state or destination.
        is_final_state = np.array_equal(self.state, self.final_state)   

        if verbose:
            print(f"_step::UPDATED FINAL STATE: {is_final_state}") 
            print(f"_step::ACTUAL STATE: {self.state}")      
            print(f"_step::Current action list: {self.actions_done}")      
           
                                  
        return self.state, reward, is_final_state       

    def print_path_episode(self):
        """
        Method that prints the path followed by the agent.

        Returns: 
            None: prints the path followed by the agent.
        """
        path = [['-' for _ in range(len(self.rewards))] for _ in range(len(self.rewards[0]))]
        path[0][0] = '0'
        for index, step in enumerate(self.actions_done):
            path[step[0]][step[1]] = str(index + 1)

        print(pd.DataFrame(data=np.array([np.array(xi) for xi in path]),
                           index=["x{}".format(str(i)) for i in range(len(path))],
                           columns=["y{}".format(str(i)) for i in range(len(path[0]))]))
                          