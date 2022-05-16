import random
from collections import defaultdict
from copy import deepcopy
from mc import MC
import numpy as np


def off_policy_mc_control(self):  # ,transformation
    robot = self.robot
    # initialization
    Q_tmp = deepcopy(self.Q)
    policy = self.policy
    D = self.D
    N = self.N
    episode_new = []
    omega = 1
    for l in range(int(self.max_iteration)):
        # generate an episode
        episode = self.generate_episode(policy)  # ,transformation
        # Choose tau with the latest time at action_tau != pi(state_tau)
        # for each pair s,a appearing in the episode t time tau or later:
        for item in episode:
            state = item[0]
            action = self.direction_index_map[item[1]]
            reward = item[2]
            for t in reversed(range(len(policy))):
                if action == policy[action, state[0], state[1]]:
                    t -= 1
                else:
                    tau = t + 1
                    for tau in range(tau, len(policy)):
                        omega *= policy[tau, state[0], state[1]]
                        R_t = reward
                        N = N + omega * R_t
                        D = D + omega
                        # calculate the new Q(s,a) by N and D
                        episode_tmp = (state[0], state[1]), self.directions[action], N / D
                        episode_new.append(episode_tmp)
            Q_tmp = self.Q_table(episode_new)
            # for each state, we find the action which argmax the Q(s,a) as the policy for this state
            for act in range(4):
                A = Q_tmp[:, state[0], state[1]]
                best_action = np.argmax(A)
                max_action_value = np.where(A == Q_tmp[best_action, state[0], state[1]])
                policy_new = [0, 0, 0, 0]
                # if the best action is not the only one, then each optimal direction has the same probability
                prob = 1 / len(max_action_value)
                for action in max_action_value:
                    for t in range(len(action)):
                        policy_new[action[t]] = prob
                policy[:, state[0], state[1]] = policy_new
    return policy


def robot_epoch(robot):
    model_free = MC(robot,gamma=1,epsilon=0.1,max_iteration=80)
    optimal_policy = off_policy_mc_control(model_free)
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
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()