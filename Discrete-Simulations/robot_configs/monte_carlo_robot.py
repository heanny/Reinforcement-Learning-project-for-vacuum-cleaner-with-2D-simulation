import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple, List, Optional
from environment import Robot


class MC:
    """
    This class implements two mode-based reinforcement learning techniques: policy based
    and value based.
    """

    def __init__(self, robot, gamma=0.1, epsilon = 0.1, max_iteration=100) -> None:
        self.robot = robot
        self.n_cols = robot.grid.n_rows
        self.n_rows = robot.grid.n_cols
        self.max_iteration = max_iteration
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = np.full((4,self.n_rows, self.n_cols),0.25)
        # self.Qvalue_table = self.init_Qvalue_table(self.n_rows, self.n_cols)
        self.directions = ['n', 'e', 's', 'w']
        self.direction_index_map = {'n': 0, 'e': 1, 's': 2, 'w': 3}
        self.trans_dirs = {(0, -1):0, (1, 0): 1, (0, 1): 2, (-1, 0):3}
        self.Q = np.zeros((4,self.n_rows,self.n_cols))

    # def init_state_value(self):
    #     state_space_cln = []
    #     state_space_obs = []
    #     for row in range(self._cells.shape[0]):
    #         for col in range(self._cells.shape[1]):
    #             if (-2 <= self._cells[row, col] < 0):
    #                 state_space_obs.append(row*col)
    #             else:
    #                 state_space_cln.append(row*col)
    #     return state_space_cln,state_space_obs

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
        print("start move")
        robot.move()
        print("end move")
        # return the new state s' and reward
        # if reward == 0:
        #     reward = transformation[robot.pos[0]][robot.pos[1]]
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
        frequency = np.zeros((robot_copy.grid.n_rows, robot_copy.grid.n_cols))
        while robot_copy.alive and np.max(robot_copy.grid.cells) > 0 and np.max(frequency) < 20:
            print("+++++++++++++++++++++++ start +++++++++++++++++++++++++++++++")
            print(robot_copy.alive, np.max(robot_copy.grid.cells))
            # current state
            state = robot_copy.pos
            print("current state", state)
            i = state[0]
            j = state[1]
            frequency[i,j] += 1
            # use policy to choose action given state
            policy_of_current_state = policy[:, i, j]
            # for a in self.directions:
            #     next_state: int = self._cells[(i + a[1]), (j + a[0])]
            #     if (-2 <= next_state < 0):
            print(policy_of_current_state)
            action = np.random.choice(self.directions, p=policy_of_current_state)

            # simulate and get s' and r
            print("start simulation")
            next_state, reward = self.simulation(robot_copy, action) #, transformation
            print("end simulation")
            print(next_state, reward)
            episode.append((state,action,reward))

        return episode

    # def make_epsilon_greedy_policy(n_rows, n_cols, Q, rewards, policy, num_actions=4, epsilon=self.epsilon):
    #     def policy_for_state(i, j):
    #         A = np.ones(num_actions, dtype=float) * epsilon / num_actions
    #         best_action = np.argmax(Q[:, i, j])
    #         A[best_action] += (1.0 - epsilon)
    #         return A
    #
    #     for i in range(0, n_rows):
    #         for j in range(0, n_cols):
    #             if rewards[i][j] not in [-3, -1]:
    #                 policy[i][j] = policy_for_state(i, j)
    #     return policy

    def on_policy_mc_control(self): # ,transformation
        robot = self.robot
        n_cols = robot.grid.n_rows
        n_rows = robot.grid.n_cols
        dirs = robot.dirs
        # initialization
        Q_tmp = deepcopy(self.Q)
        policy = self.policy
        print("policy:")
        print(policy)
        # first set the policy to be zero
        # policy = self.make_epsilon_greedy_policy(n_rows, n_cols, Q, rewards, policy, num_actions=4, epsilon=0.1)
        epsilon = self.epsilon
        # repeat till value converge:
        for l in range(int(self.max_iteration)):
            # generate an episode
            episode = self.generate_episode(policy) #,transformation
            # Update Q table for each (s,a) in episode
            Q_tmp = self.Q_table(episode)
            # TODO: Change |A(s)|
            for item in episode: # (state,action,reward)
                state=item[0]
                best_action = np.argmax(Q_tmp[:,state[0],state[1]])
                # self.direction_index_map.items()
                for action in range(4): 
                    # self.robot.possible_tiles_after_move()
                    # a = self.trans_dirs[action]
                    if action == best_action:
                        policy[action,state[0],state[1]] = 1-epsilon+epsilon/4
                    else:
                        policy[action,state[0],state[1]] = epsilon/4
        return policy




def robot_epoch(robot):
    model_free = MC(robot,gamma=0.2,epsilon=0.2,max_iteration=10)
    optimal_policy = model_free.on_policy_mc_control()
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    for index in range(0, 4):
        if index in indices:
            probability.append(1/len(indices))
        else:
            probability.append(0)
    direction = np.random.choice(model_free.directions, p=probability)
    print("+++++++++++++++++++++++ move +++++++++++++++++++++++++++++++")
    print(direction)
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()