from collections import defaultdict
from copy import deepcopy
import numpy as np

class MC:
    """
    This class implements two mode-based reinforcement learning techniques: policy based
    and value based.
    """

    def __init__(self, robot, gamma=1, epsilon = 0.1, max_iteration=100) -> None:
        self.robot = robot
        self.n_cols = robot.grid.n_rows
        self.n_rows = robot.grid.n_cols
        self.max_iteration = max_iteration
        self.gamma = gamma
        self.epsilon = epsilon
        self.N = 0  # numerator of Q(s,a)
        self.D = 0  # denominator of Q(s,a)
        self.policy = np.full((4,self.n_rows, self.n_cols),0.25)
        # self.Qvalue_table = self.init_Qvalue_table(self.n_rows, self.n_cols)
        self.directions = ['n', 'e', 's', 'w']
        self.direction_index_map = {'n': 0, 'e': 1, 's': 2, 'w': 3}
        # self.trans_dirs = {(0, -1):0, (1, 0): 1, (0, 1): 2, (-1, 0):3}
        self.trans_dirs = {(-1, 0):0, (0, 1): 1, (1, 0): 2, (0, -1):3}
        self.Q = np.zeros((4,self.n_rows,self.n_cols))


    def simulation(self, robot, action): # , transformation
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
        robot.move()
        return robot.pos, reward

    def Q_table(self, episode):
        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)
        sa_in_episode = set([(x[0], x[1]) for x in episode])
        Q = self.Q
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            action_num = self.direction_index_map[action]
            # Calculate Q(s,a) for each (s,a) pair (mc policy evaluation)
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            # Sum up all rewards since the first occurance
            G = sum([x[2] * (self.gamma ** i) for i, x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[action_num][state[0]][state[1]] = returns_sum[sa_pair] / returns_count[sa_pair]
        return Q


    def generate_episode(self,policy): #,transformation

        episode = []
        robot_copy = deepcopy(self.robot)
        frequency = np.zeros((robot_copy.grid.n_cols, robot_copy.grid.n_rows))
        while robot_copy.alive and np.max(robot_copy.grid.cells) > 0 and np.max(frequency) < 3:
            # current state
            state = robot_copy.pos
            i = state[0]
            j = state[1]
            frequency[i,j] += 1
            # use policy to choose action given state
            policy_of_current_state = policy[:, i, j]

            #can we not use the policy_of_current_state for p in action choice?
            action = np.random.choice(self.directions, p=policy_of_current_state)

            # simulate and get s' and r
            next_state, reward = self.simulation(robot_copy, action)

            episode.append((state,action,reward))
        return episode
