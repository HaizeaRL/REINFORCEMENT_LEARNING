import numpy as np
import pandas as pd

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
        self.state = src                             # Agent current state. Initialize in source position.
        self.final_state = dest                      # Agents destination. When the agent arrives, the episode ends.
        self.total_reward = 0.0                      # Episode reward counter
        self.actions_done = []                       # List of steps (actions) performed in each episode. 

    def reset(self, src):
        """
        Method that resets the environment variables and returns the initial state.

        Parameters:
            src (tuple): Start position in the maze.

        Returns:    
            initial position of the maze.
        """
        self.total_reward = 0.0    # Initialize episode reward counter
        self.state = src           # Positionate the agent in the initial state
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
            action: Action to be taken.

        Returns:
            current state
        """
        self.state[0] += self.actions[action][0]
        self.state[1] += self.actions[action][1]

        # control boundaries.
        self.state[0], self.state[1] = self.stay_within_bounds()

    def step(self, action):
        """
        Method that executes a given action from the set of actions {Up, Down, Left, Right}
        Guide the agent in the environment.

        Parameters:
            action: Action to be taken.

        Returns: 
            self.state (tuple): The current state
            reward (int): The obtained reward
            is_final_state (boolean): Whether we have reached the final state.
        """
        self.__apply_action(action)                                                # Perform the action (change of state)
        self.actions_done.append(self.state[:])                                    # Save the action or step taken.
        is_final_state = np.array_equal(self.state, self.final_state)              # Check if we are in the final state or destination.
        reward = self.rewards[self.state[0]][self.state[1]] + self.action_penalty  # Calculate the reward (reward) for the action taken
        self.total_reward += reward                                                # Add the current reward to the total reward for the episode
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
                          