from environment import Robot
import numpy as np
import copy
import random
from numpy.random import choice

class ModelFree:
    """
    This class implements model free reinforcement learning techniques.
    """

    def __init__(self, robot: Robot) -> None:
        self.robot = robot
        self.n_cols = robot.grid.n_rows
        self.n_rows = robot.grid.n_cols
        self.policy = self.init_policy(self.n_rows, self.n_cols)
        self.Qvalue_table = self.init_Qvalue_table(self.n_rows, self.n_cols)
        self.directions = ['n', 'e', 's', 'w']
        self.direction_index_map = {'n': 0, 'e': 1, 's': 2, 'w': 3}

    # initialization function
    def init_policy(self, n_rows, n_cols):
        """
        Initialize the policy matrix, where each element is a dictionary that shows
        the probability of moving in a certain direction in a given state.
        We initialize each direction with a probability of 1 in 4.
        :param n_rows: number of rows in the grid
        :param n_cols: number of columns in the grid
        :returns policy: the policy 3D matrix
        """
        initial_pobability = 0.25
        policy = np.full((4, n_rows, n_cols), initial_pobability)
        return policy

    def init_Qvalue_table(self, n_rows, n_cols):
        """
        Initialize the Q-value table, where each element is a dictionary that shows
        the value of moving in a certain direction in a given state.
        We initialize every state-action pair as 0.
        :param n_rows: number of rows in the grid
        :param n_cols: number of columns in the grid
        :returns policy: the 3D Q-value matrix
        """
        return np.zeros((4, n_rows, n_cols))

    def simulation(self, robot, action):
        # get reward of action
        coordinate = robot.dirs[action]
        possible_tiles = robot.possible_tiles_after_move()
        reward = possible_tiles[coordinate]
        if reward == 3:
            reward = -2
        if reward == -2:
            reward = -1
        # take action
        while not action == robot.orientation:
            # If we don't have the wanted orientation, rotate clockwise until we do:
            robot.rotate('r')
        print("start move")
        robot.move()
        print("end move")
        # return the new state s' and reward
        # if reward == 0:
            # reward = transformation[robot.pos[0]][robot.pos[1]]
        return robot.pos, reward

    def update_Qvalue(self, action, state, next_state, reward, alpha, gamma, on_policy, next_action):
        # update Qvalue table
        action_index = self.direction_index_map[action]
        old_Qvalue = self.Qvalue_table[action_index, state[0], state[1]]  # get Q(s,a)
        print("old Qvalue:", old_Qvalue)
 # get max Qvalue of s'

        if on_policy:
            next_action_Qvalues = self.Qvalue_table[next_action, next_state[0], next_state[1]]
            self.Qvalue_table[action_index, state[0], state[1]] = old_Qvalue + alpha * (
                        reward + gamma * next_action_Qvalues - old_Qvalue)
        else:
            next_state_Qvalues = self.Qvalue_table[:, next_state[0], next_state[1]]  # get all the Q(s',a)
            next_state_max_Qvalue = max(next_state_Qvalues)
            self.Qvalue_table[action_index, state[0], state[1]] = old_Qvalue + alpha * (reward + gamma * next_state_max_Qvalue - old_Qvalue)
        print("new Qvalue:", self.Qvalue_table[action_index, state[0], state[1]])

    def update_policy(self, epsilon, state):
        # update epsilon-greedy policy
        print("old policy:", self.policy[:, state[0], state[1]])
        old_policy = self.policy[:, state[0], state[1]]
        Qvalues = self.Qvalue_table[:, state[0], state[1]]  # get current state all Qvalues
        max_Qvalue = max(Qvalues)  # get the highest Q(s,a) for s, there could be more than 1 highest Q(s,a)
        indices = [index for index, value in enumerate(Qvalues) if
                   value == max_Qvalue]  # find the indices of all max value
        smallest_probability = epsilon / 4  # smallest_probability for maintaining exploration
        greedy_probability = (1 - epsilon) / len(indices) + epsilon / 4
        for index in range(0, 4):
            if index in indices:
                self.policy[index, state[0], state[1]] = greedy_probability
            else:
                self.policy[index, state[0], state[1]] = smallest_probability
        print("new policy:", self.policy[index, state[0], state[1]])