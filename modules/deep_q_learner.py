'''
DEEP Q-LEARNING FUNCTIONS
'''

import numpy as np
import random
import pandas as pd
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class DeepQLearner(object):
   
    def __init__(self, environment, max_memory=100, discount_factor=0.1, explotation_rate=0.95):
        """
        Class implementing a reinforcement Deep Q-learning algorithm.
        
        Parameters:
            environment (Environment): The environment in which the algorithm will interact and take actions.
            max_memory (int):  Maximum number of actions to be memorized (saved) in one episode
            discount_factor (float): Discount factor to weigh future rewards. 
                (0 = short-term focus, 1 = long-term focus).
            explotation_rate (float): Explotation ratio, controlling the trade-off between exploration and exploitation.
        """
        self.environment = environment       
        self.model = self.create_model()   # Neural Network creation
        self.discount_factor = discount_factor # Short-term or long-term 

        # Experience Replay controls:  Current State | Taken Action | Reward | Next State | Is final state?
        self.memory = list()  # List of actions taken                  
        self.max_memory = max_memory # maximum memory size
        
        # Too ratios: these values will gradually guide the agent from exploration to exploitation.
        self.max_explotation_rate = explotation_rate
        self.explotation_rate = 0 
       

    
    @property
    def name(self):
        return 'Deep Q-Learner'
    
    def create_model(self):
        """
        Function that creates corresponding neuronal network.

        Returns:
            Neuronal network.
        :return: Red Neuronal
        """

        input_dim = len(self.environment.state)      # Number of neurons of input layer '2'. (X,Y)
        output_dim = len(self.environment.actions)   # Number of neurons of ouput layer '4'. (Up, Down, Right, Left)

        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=input_dim))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(loss='mse', optimizer='adam')

        return model
    
    def get_next_action(self, state):
        """
        Method that selects the next action to take:
            Aleatoria -> if the exploration ratio is below the threshold.
            Best Action ->  if the exploration ratio is higher than the threshold.

        Parameters:
            state: Agent state

        Returns:
            next_action: Agent next action.
        """

        if np.random.uniform() > self.explotation_rate:
            # selection of possible actions at random.
            next_action = np.random.choice(list(self.environment.actions))
        else:
            # selection of the action that gives the highest value.
            qus = self.model.predict([state], verbose=0)
            idx_action = np.argmax(qus[0])
            next_action = list(self.environment.actions)[idx_action]

        return next_action
    
    def reset(self):
        """
        Function that reset agent's experience replay or memory.
        
        Returns:
            An empty list of memory.
        """
        del self.memory[:]

    
    def save_experience_replay_in_memory(self, state, action, reward, new_state, is_final_state):
        """
        Function that stores in a list, a tuple with information of each of the steps performed by the agent 
        in the environment during the episode. Stored data:
        - the **current state**,
        - the **action taken**,
        - the **reward** obtained,
        - the **next state**, and
        - whether or not **it is a terminal state**.

        In case the number of actions in the memory is higher than the maximum number of actions to save, 
        we will delete the oldest action from the list.

        Parameters:
            state: Agent state
            action: Taken action by the agent.
            reward: Reward obtained after make the action.
            new_state: The new state the agent is moved after take the action.
            is_final_state (Boolean): Flag which indicate whether the agent reach to destination or not.
        """
        # append experience_replay to memory
        self.memory.append((state, action, reward, new_state, is_final_state))
        # if reach to limit remove first action.
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def learn(self, actions, num_episode):
        """
        Function that updates the model (Neural Network) - Learns from the actions performed in the episode.
        Gets randomly some actions from memory and apply weight adjust.

        This method also balance exploring and explotation actions by updating the exploitation ratio as follows:
        exploitation_ratio = exploitation_ratio - (max_exploitation_ratio / (num_episodes + 1))

        Parameters:
            actions: Possible agent actions {Up, Down, Right, Left} to get taken action index to update q_value.
            num_episode: Episode number, used to update gradually the exploration rate.

        Returns:
            None: Learns from actions taken based on learned knowledge as episodes progress.        
        """
        # select randomly some steps from memory.
        batch = (random.sample(self.memory, 100)
                 if len(self.memory) > 100 else random.sample(self.memory, len(self.memory)))

        # for each selected steps, apply the learning process
        for state, action, reward, new_state, is_final_state in batch:
            # Predict Q-value for state
            q_values = self.model.predict([state], verbose=0)
            idx_action = list(actions).index(action)

            # Calculate objetive best Q-value for the action
            q_values[0][idx_action] = (reward + (self.discount_factor * np.amax(self.model.predict([new_state], verbose=0)[0]))
                                       if not is_final_state else reward)

            # Adjust weights to obtain objective Q-value.
            self.model.fit(np.array([state]), q_values, epochs=1, verbose=0)

        # Update explotation rate gradually to guide the agent from exploration to exploitation.
        self.explotation_rate = self.max_explotation_rate - (self.max_explotation_rate / (num_episode + 1))

    
    def update(self, environment, state, action, reward, new_state, is_final_state, num_episode):
        """
        Function that implements Deep Q-Learning reinforcement learning 

        Parameters:
            environment (Environment): The environment in which the algorithm will interact and take actions.
            state: The agent's current state.
            action: The action to be taken.
            reward: The reward received for taking the specified action.
            new_state: The new state reached by the agent after taking the action.
            is_final_state: Boolean indicating whether the agent has reached the terminal state.
            num_episode (int): Number of executed episodes.

        Returns:
            None: Updates the Q-table in place according to the Q-Learning algorithm.
        """
        # save current state in memory
        self.save_experience_replay_in_memory(state=state, action=action, reward=reward, new_state=new_state,
                                               is_final_state=is_final_state)
        # check whether reached to the episode end or final state. If yes, train the model for learning.
        if is_final_state: 
            print(f"Starting learning process...")
            self.learn(actions=environment.actions, num_episode=num_episode)
            self.reset()

    def print_q_table(self):
        """
        Function that prints Q-Values learned by the NN.
        """
        # Get all possible states. 
        states = list(itertools.product(list(range(0, len(self.environment.rewards[0]))), repeat=2))

        # Predict all states Q(s,a)
        q_table = self.model.predict(states, verbose=0)    

        # Print the results in a dataframe form
        df = (pd.DataFrame(data=q_table,                          
                           columns=['Up', 'Down', 'Left', 'Right']))
        
        # add state to action in first place. Example: x0,y0 actions_values.
        df.insert(0, 'State', ['x{},y{}'.format(state[0], state[1]) for state in states])

        # visualize the result
        print(df.to_string(index=False))

    
    def print_best_actions_states(self):
        """
        Function that prints best value of the best option to be performed in each of the states.
        """
        # Get all possible states. 
        states = list(itertools.product(list(range(0, len(self.environment.rewards[0]))), repeat=2))

        # Predict all states Q(s,a)
        q_table = self.model.predict(states, verbose=0)     

        # Obtain a list of best actions for each state based on the highest Q-value.
        best_actions = np.array([list(self.environment.actions)[np.argmax(row)] for row in q_table])

        # Reshape best actions into a matrix with the same dimensions as the environment.
        best = (best_actions.reshape(len(self.environment.rewards), # rows dimension
                                     len(self.environment.rewards[0]))) # cols dimension

        # print result as dataframe
        print(pd.DataFrame(data=np.array([np.array(xi) for xi in best]), # Place best action values in each state
                           index=["x{}".format(str(i)) for i in range(len(best))],  # Add row labels
                           columns=["y{}".format(str(i)) for i in range(len(best[0]))])) # Add col labels

    def print_best_values_states(self):
        """
        Método que imprime por pantalla el valor de la mejor opción a realizar en cada uno de los estados
        """
        # Get all possible states. 
        states = list(itertools.product(list(range(0, len(self.environment.rewards[0]))), repeat=2))

        # Predict all states Q(s,a)
        q_table = self.model.predict(states, verbose=0)     

        # obtain best q values of each state 
        best_values = np.array([[np.max(row) for row in q_table]])

        # Reshape best values into a matrix with the same dimensions as the environment.
        best = best_values.reshape(len(self.environment.rewards), # rows dimension
                                    len(self.environment.rewards[0]))  # cols dimension

        print(pd.DataFrame(data=np.array([np.array(xi) for xi in best]),  # Place best q values in each state
                           index=["x{}".format(str(i)) for i in range(len(best))], # Add row labels
                           columns=["y{}".format(str(i)) for i in range(len(best[0]))])) # Add col labels

