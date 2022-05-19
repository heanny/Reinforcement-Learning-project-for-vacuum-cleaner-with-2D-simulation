from collections import defaultdict
from copy import deepcopy
import numpy as np

class MC:
    """
    This class implements the on-policy and off-policy Monte Carlo methods.
    """

    def __init__(self, robot, gamma=1, epsilon = 0.1, max_iteration=100) -> None:
        self.robot = robot
        self.n_cols = robot.grid.n_rows # switch the columns and rows to correctly match the grid
        self.n_rows = robot.grid.n_cols
        self.max_iteration = max_iteration
        self.gamma = gamma
        self.epsilon = epsilon
        self.N = 0  # numerator of Q(s,a)
        self.D = 0  # denominator of Q(s,a)
        self.policy = np.full((4,self.n_rows, self.n_cols),0.25) # initializing the policy matrix, which corresponds to epsilon-soft policy
        self.directions = ['n', 'e', 's', 'w']
        self.direction_index_map = {'n': 0, 'e': 1, 's': 2, 'w': 3} # list index corresponds to direction name
        self.trans_dirs = {(-1, 0):0, (0, 1): 1, (1, 0): 2, (0, -1):3} # list index corrrespond to direction coordinate
        self.Q = np.zeros((4,self.n_rows,self.n_cols)) # initializing Q table
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)


    def simulation(self, robot, action): 
        """
        Simulate an action of the episode and give its corresponding reward.

        :param robot: the robot copy
        :param action: the proposed action
        :return robot.pos: the position of robot
        :return reward: the reward for the action
        """
        # get reward of action
        coordinate = robot.dirs[action]
        possible_tiles = robot.possible_tiles_after_move()
        reward = possible_tiles[coordinate]
        # give reward
        if reward == 3: # death tile, give -2 to avoid robot reach it.
            reward = -2
        if reward == -2: # an obstacle, we think they have the same function of wall tiles, so we reset as -1
            reward = -1
        # take action
        while not action == robot.orientation:
            # If we don't have the wanted orientation, rotate clockwise until we do:
            robot.rotate('r')
        robot.move()
        return robot.pos, reward


    def generate_episode(self,policy):
        """
        Generate an episode based on a policy.

        :param policy: the current policy
        :return episode: the generated episode (list of (state, action, reward))
        """
        episode = []
        robot_copy = deepcopy(self.robot)
        # frequency: the number of times that the tile is visited
        frequency = np.zeros((robot_copy.grid.n_cols, robot_copy.grid.n_rows))
        # condition: robot is alive; not terminated; and the tile is visited no more than three times
        while robot_copy.alive and np.max(robot_copy.grid.cells) > 0 and np.max(frequency) < 3:
            # current state
            state = robot_copy.pos
            i = state[0]
            j = state[1]
            frequency[i,j] += 1
            # given state, use policy to choose action
            policy_of_current_state = policy[:, i, j]
            action = np.random.choice(self.directions, p=policy_of_current_state)

            # simulate and get s' and r
            next_state, reward = self.simulation(robot_copy, action)

            episode.append((state,action,reward))
        return episode

    def Q_table(self, episode):
        """
        Generate Q_table based on athe episode.

        :param episode: an episode (list of (state, action, reward))
        :return Q: the updated Q table according to the episode
        """
        # find unique state and action in an episode
        sa_in_episode = set([(x[0], x[1]) for x in episode])
        Q = self.Q
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            action_num = self.direction_index_map[action] # get index for action
            # Calculate Q(s,a) for each (s,a) pair (mc policy evaluation)
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            # Sum up all rewards since the first occurance
            G = sum([x[2] * (self.gamma ** i) for i, x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all the episodes
            self.returns_sum[sa_pair] += G
            self.returns_count[sa_pair] += 1.0
            Q[action_num][state[0]][state[1]] = self.returns_sum[sa_pair] / self.returns_count[sa_pair]
        return Q