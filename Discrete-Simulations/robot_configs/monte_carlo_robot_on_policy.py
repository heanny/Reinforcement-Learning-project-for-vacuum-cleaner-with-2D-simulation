import random
from collections import defaultdict
from copy import deepcopy
import numpy as np
from mc import MC


def on_policy_mc_control(self):  # ,transformation
    robot = self.robot
    n_cols = robot.grid.n_rows
    n_rows = robot.grid.n_cols
    dirs = robot.dirs
    # initialization
    Q_tmp = deepcopy(self.Q)
    policy = self.policy
    # first set the policy to be zero
    epsilon = self.epsilon
    # repeat till value converge:
    for l in range(int(self.max_iteration)):
        # generate an episode
        episode = self.generate_episode(policy)  # ,transformation
        # Update Q table for each (s,a) in episode
        Q_tmp = self.Q_table(episode)

        for item in episode:  # (state,action,reward)
            state = item[0]
            A = Q_tmp[:, state[0], state[1]]
            best_action = np.argmax(A)
            best_action_indices = np.where(A == Q_tmp[best_action, state[0], state[1]])
            for action in range(4):
                if action in best_action_indices[0]:
                    policy[action, state[0], state[1]] = (1 - epsilon) / len(best_action_indices[0]) + epsilon / 4
                else:
                    policy[action, state[0], state[1]] = epsilon / 4

    return policy


def robot_epoch(robot):
    model_free = MC(robot,gamma=0.3,epsilon=0.2,max_iteration=10)
    optimal_policy = on_policy_mc_control(model_free)
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
    # print("+++++++++++++++++++++++ move +++++++++++++++++++++++++++++++")
    # print(direction)
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()

def robot_epoch_(robot,gamma,epsilon):
    model_free = MC(robot,gamma,epsilon,max_iteration=200)
    optimal_policy = on_policy_mc_control(model_free)
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