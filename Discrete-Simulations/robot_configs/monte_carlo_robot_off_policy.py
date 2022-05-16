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
        # print("start move")
        robot.move()
       # print("end move")
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
        # print("generate_episode_policy")
        # print(policy)
        episode = []
        robot_copy = deepcopy(self.robot)
        frequency = np.zeros((robot_copy.grid.n_cols, robot_copy.grid.n_rows))
        while robot_copy.alive and np.max(robot_copy.grid.cells) > 0 and np.max(frequency) < 3:
            #print("+++++++++++++++++++++++ start +++++++++++++++++++++++++++++++")
            #print(robot_copy.alive, np.max(robot_copy.grid.cells))
            # current state
            state = robot_copy.pos
            #print("current state", state)
            i = state[0]
            j = state[1]
            frequency[i,j] += 1
            # use policy to choose action given state
            policy_of_current_state = policy[:, i, j]
            # print("policy_of_current_state")
            # print(policy_of_current_state)
            # for a in self.directions:
            #     next_state: int = self._cells[(i + a[1]), (j + a[0])]
            #     if (-2 <= next_state < 0):
            # print(policy_of_current_state)

            #can we not use the policy_of_current_state for p in action choice?
            action = np.random.choice(self.directions)
            #action = np.random.choice(self.directions, p=policy_of_current_state)

            # simulate and get s' and r
            #print("start simulation")
            next_state, reward = self.simulation(robot_copy, action) #, transformation
            #print("end simulation")
            #print(next_state, reward)
            episode.append((state,action,reward))
        # print("episode")
        # print(episode)
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
        # first set the policy to be zero
        # policy = self.make_epsilon_greedy_policy(n_rows, n_cols, Q, rewards, policy, num_actions=4, epsilon=0.1)
        epsilon = self.epsilon
        # repeat till value converge:
        for l in range(int(self.max_iteration)):
            # generate an episode
            episode = self.generate_episode(policy) #,transformation
            # Update Q table for each (s,a) in episode
            Q_tmp = self.Q_table(episode)
            # print("Q_tem:")
            # print(Q_tmp)
            # TODO: Change |A(s)|
            for item in episode: # (state,action,reward)
                state=item[0]
                A=Q_tmp[:,state[0],state[1]]
                best_action = np.argmax(A)
                best_action_indices = np.where(A==Q_tmp[best_action,state[0],state[1]])
                # print("-------length and value of best actions-----")
                # print(len(best_action_indices),best_action_indices[0])
                # self.direction_index_map.items()
                for action in range(4): 
                    # self.robot.possible_tiles_after_move()
                    # a = self.trans_dirs[action]
                    if action in best_action_indices[0]:
                        policy[action,state[0],state[1]] = (1-epsilon)/len(best_action_indices[0])+epsilon/4
                    else:
                        policy[action,state[0],state[1]] = epsilon/4
            # print("policy for state and action")
            # print(policy[action,state[0],state[1]])
        # print("policy matrix:")
        # print(policy)
        return policy



    def off_policy_mc_control(self):# ,transformation
        robot = self.robot
        n_cols = robot.grid.n_rows
        n_rows = robot.grid.n_cols
        dirs = robot.dirs
        # initialization
        Q_tmp = deepcopy(self.Q)
        policy = self.policy
        # print("initial_policy")
        # print(policy)
        D = self.D
        N = self.N
        # first set the policy to be zero
        # policy = self.make_epsilon_greedy_policy(n_rows, n_cols, Q, rewards, policy, num_actions=4, epsilon=0.1)
        epsilon = self.epsilon
        #reward = []
        episode_new = []
        omega = 1
        #d = [0.25, 0.25, 0.25,0.25]
        # print("optimal_policy_ini")
        #
        # optimal_policy = np.full((n_rows, n_cols), 0.25)
        # print(optimal_policy)
        # repeat until value converge:
        for l in range(int(self.max_iteration)):
            # policy_new = {'n': 0, 'e': 0, 's': 0, 'w': 0}
            # generate an episode
            episode = self.generate_episode(policy) #,transformation

            # Choose tau with the latest time at action_tau != pi(state_tau)
            # for each pair s,a appearing in the episode t time tau or later:
            for item in episode:
                state = item[0]
                action = self.direction_index_map[item[1]]
                reward = item[2]
                # print("action")
                # print(action)
                #reward.append(item[2])
                #print("reward")
                #print(reward)
                for t in reversed(range(len(policy))):
                    if action == policy[action, state[0], state[1]]:
                        t -= 1
                    else:
                        tau = t+1
                        for tau in range(tau, len(policy)):
                            omega *= policy[tau, state[0], state[1]]
                            R_t = reward
                            N = N + omega * R_t
                            D = D + omega
                            # print(self.directions[action])
                            episode_tmp = (state[0], state[1]), self.directions[action], N/D
                            episode_new.append(episode_tmp)
                Q_tmp = self.Q_table(episode_new)
                # print('Q_tmp')
                # print(Q_tmp)
                for act in range(4):
                    A = Q_tmp[:, state[0], state[1]]
                    # print('A')
                    # print(A)
                    best_action = np.argmax(A)
                    # print('best_action')
                    # print(best_action)
                    #prob = 1/len(A)
                    max_action_value = np.where(A == Q_tmp[best_action, state[0], state[1]])
                    # print('max_action_value')
                    # print(max_action_value)
                    # max_action_value = [key for m in [max(Q_tmp.values())] for key, val in Q_tmp.items() if val == m]
                #new_policy = {'n': 0, 'e': 0, 's': 0, 'w': 0}
                #policy_new = {'n': 0, 'e': 0, 's': 0, 'w': 0}
                    policy_new = [0,0,0,0]
                    prob = 1 / len(max_action_value)
                    for action in max_action_value:
                    # print('action')
                    # print(action)
                        for t in range(len(action)):
                            policy_new[action[t]] = prob
                # print('policy_new')
                # print(policy_new)
                    policy[:,state[0],state[1]] = policy_new
        # print("optimal_policy")
        # print(policy)
        return policy

def robot_epoch(robot):
    model_free = MC(robot,gamma=1,epsilon=0.1,max_iteration=80)
    optimal_policy = model_free.off_policy_mc_control()
    # print(optimal_policy)
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    for index in range(0, 4):
        if index in indices:
            probability.append(1/len(indices))
        else:
            probability.append(0)
    direction = np.random.choice(model_free.directions, p=probability)
    #print("+++++++++++++++++++++++ move +++++++++++++++++++++++++++++++")
   # print(direction)
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()