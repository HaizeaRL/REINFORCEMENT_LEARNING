'''
DEEP Q-LEARNING FUNCTIONS
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import random

class DeepQLearner(object):
    def __init__(self, environment, max_memory=100, discount_factor=0.1, explotation_rate=0.95, max_steps=500):
        """
        Class implementing a reinforcement Deep Q-learning algorithm.
        
        Parameters:
            environment (Environment): The environment in which the algorithm will interact and take actions.
            max_memory (int):  Maximum number of actions to be memorized (saved) in one episode
            discount_factor (float): Discount factor to weigh future rewards. 
                (0 = short-term focus, 1 = long-term focus).
            explotation_rate (float): Explotation ratio, controlling the trade-off between exploration and exploitation.
            max_steps (int): Maximum number of steps to take per each episode.
        """
        self.environment = environment
        # Experience Replay:  Current State | Taken Action | Reward | Next State | Is final state?
        self.memory = list()               
        self.max_memory = max_memory
        self.model = self.create_model()   # Neural Network creation
        self.discount_factor = discount_factor
        self.max_explotation_rate = explotation_rate
        self.explotation_rate = 0
        self.max_steps= max_steps

    
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
            qus = self.model.predict([state])
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

    def learn(self, environment, num_episode):
        """
        TODO: COMMENT CODE AN LEARN HOW IT WORKS
        Método que actualiza el modelo (Red Neuronal) - Aprende de las acciones realizadas en el episodio.
        Este método también actualiza el ratio de explotación de las siguiente manera:
        ration_explotacion = ratio_explotación - (maximo_ratio_explotacion / (num_episodios + 1))
        :param environment:       Entorno en el que tomar las acciones
        :param num_episode:       Número del episodio
        """
        batch = (random.sample(self.memory, 100)
                 if len(self.memory) > 100 else random.sample(self.memory, len(self.memory)))

        for state, action, reward, new_state, is_final_state in batch:
            q_values = self.model.predict([state])
            idx_action = list(environment.actions).index(action)

            q_values[0][idx_action] = (reward + (self.discount_factor * np.amax(self.model.predict([new_state])[0]))
                                       if not is_final_state else reward)

            self.model.fit(np.array([state]), q_values, epochs=1, verbose=0)

        # Actualizo el ratio de explotación
        self.explotation_rate = self.max_explotation_rate - (self.max_explotation_rate / (num_episode + 1))

    
    def update(self, environment, state, action, reward, new_state, is_final_state, num_episode, num_steps):
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
            num_steps (int): Number of steps taken in the episode.

        Returns:
            None: Updates the Q-table in place according to the Q-Learning algorithm.
        """
        # save current state in memory
        self.save_experience_replay_in_memory(state=state, action=action, reward=reward, new_state=new_state,
                                               is_final_state=is_final_state)
        # check whether reached to the episode end or final state. If yes, train the model for learning.
        if is_final_state or num_steps > self.max_steps:
            self.learn(environment=environment, num_episode=num_episode)
            self.reset()

    '''
    TODO: COMMENT CODE AN LEARN HOW IT WORKS
    def print_q_table(self):
        """
        Método que imprime por pantalla la Q-Table aprendida por la red
        """
        states = list(itertools.product([0, 1, 2, 3], repeat=2))  # Generamos todos los posibles estados
        q_table = self.model.predict(states)                      # Predecimos con la red los Q(s,a)
        df = (pd.DataFrame(data=q_table,                          # Pasamos la Q_Table a un DataFrame
                           columns=['Arriba', 'Abajo', 'Izquierda', 'Derecha']))
        df.insert(0, 'Estado', ['x{},y{}'.format(state[0], state[1]) for state in states])
        print(df.to_string(index=False))

    def print_best_actions_states(self):
        """
        Método que imprime por pantalla el valor de la mejor opción a realizar en cada uno de los estados
        """
        states = list(itertools.product([0, 1, 2, 3], repeat=2))  # Generamos todos los posibles estados
        q_table = self.model.predict(states)  # Predecimos con la red los Q(s,a)

        best = (np.array([list(self.environment.actions)[np.argmax(row)] for row in q_table])
                .reshape(len(self.environment.rewards), len(self.environment.rewards[0])))

        print(pd.DataFrame(data=np.array([np.array(xi) for xi in best]),
                           index=["x{}".format(str(i)) for i in range(len(best))],
                           columns=["y{}".format(str(i)) for i in range(len(best[0]))]))

    def print_best_values_states(self):
        """
        Método que imprime por pantalla el valor de la mejor opción a realizar en cada uno de los estados
        """
        states = list(itertools.product([0, 1, 2, 3], repeat=2))  # Generamos todos los posibles estados
        q_table = self.model.predict(states)                      # Predecimos con la red los Q(s,a)

        best = (np.array([[np.max(row) for row in q_table]])
                .reshape(len(self.environment.rewards), len(self.environment.rewards[0])))

        print(pd.DataFrame(data=np.array([np.array(xi) for xi in best]),
                           index=["x{}".format(str(i)) for i in range(len(best))],
                           columns=["y{}".format(str(i)) for i in range(len(best[0]))]))'''

