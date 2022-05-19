from environment import Robot
import numpy as np

class TD:
    """
    This class implements temporal difference algorithms: SARSA, Q-learning.

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
        """
        Simulate an action of the episode and give its corresponding position and reward.

        :param robot: the robot copy
        :param action: the proposed action
        :return robot.pos: the position of robot
        :return reward: the reward for the action
        """
        coordinate = robot.dirs[action]
        possible_tiles = robot.possible_tiles_after_move()
        # get the reward of an action
        reward = possible_tiles[coordinate]
        # the death tile has reward of -3
        if reward == 3:
            reward = -3
        # the wall and obstacle tiles have reward of -1
        if reward == -2:
            reward = -1
        # the cleaned tiles have reward of 0
        if reward == 0:
            reward = 0
        # the goal and dirty tiles have reward of 1
        if 3 > reward >= 1:
            reward = 1
        # take action
        while not action == robot.orientation:
            # If we don't have the wanted orientation, rotate clockwise until we do:
            robot.rotate('r')
        robot.move()
        # return the new state s' and reward
        return robot.pos, reward

    def update_Qvalue(self, action, state, next_state, reward, alpha, gamma, on_policy, next_action):
        """
        Update the value in Q table based on given arguments

        :param action: the current action
        :param state: the current state
        :param next_state: the next state
        :param reward: the reward of the current action
        :param alpha: the learning rate
        :param gamma: the discounted factor
        :param alpha: the learning rate
        :param on_policy: the flag for control the Q-learning (on-policy: False) and SARSA (on-policy: True)
        :param next_action: the next action
        :return robot.pos: the position of robot
        :return reward: the reward for the action
        """
        # update Qvalue table
        action_index = self.direction_index_map[action]
        # get the old value Q(s,a)
        old_Qvalue = self.Qvalue_table[action_index, state[0], state[1]]  # get Q(s,a)
        # get max Q-value of s'
        # if it is SARSA algorithm
        if on_policy:
            # if it is SARSA algorithm
            next_action_index = self.direction_index_map[next_action]
            # calculate the Q(s',a)
            next_action_Qvalues = self.Qvalue_table[next_action_index, next_state[0], next_state[1]]
            self.Qvalue_table[action_index, state[0], state[1]] = old_Qvalue + alpha * (reward + gamma * next_action_Qvalues - old_Qvalue)

        else:
            # if it is Q-learning algorithm
            next_state_Qvalues = self.Qvalue_table[:, next_state[0], next_state[1]]
            # calculate the Q(s',a)
            next_state_max_Qvalue = max(next_state_Qvalues)
            self.Qvalue_table[action_index, state[0], state[1]] = old_Qvalue + alpha * (reward + gamma * next_state_max_Qvalue - old_Qvalue)

    def update_policy(self, epsilon, state):
        """
        Update the epsilon-greedy policy based on given arguments epsilon and current state

        :param epsilon:
        :param state: the current state
        """
        # get current state all Q-values
        Qvalues = self.Qvalue_table[:, state[0], state[1]]

        # get the highest Q(s,a) for s, there could be more than 1 highest Q(s,a)
        max_Qvalue = max(Qvalues)
        # find the indices of all max values
        indices = [index for index, value in enumerate(Qvalues) if value == max_Qvalue]
        # smallest_probability for maintaining exploration
        smallest_probability = epsilon / 4
        greedy_probability = (1 - epsilon) / len(indices) + epsilon / 4
        for index in range(0, 4):
            if index in indices:
                self.policy[index, state[0], state[1]] = greedy_probability
            else:
                self.policy[index, state[0], state[1]] = smallest_probability