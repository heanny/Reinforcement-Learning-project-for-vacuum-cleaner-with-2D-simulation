import copy
import numpy as np
from numpy.random import choice
from td import TD


def sarsa(model_free, alpha, gamma, epsilon, episodes):
    while episodes:
        count = np.zeros((model_free.n_rows, model_free.n_cols))
        robot_copy = copy.deepcopy(model_free.robot)
        done = False
        # initial state
        state = robot_copy.pos
        policy = model_free.policy[:, state[0], state[1]]
        action = choice(model_free.directions, p=policy)
        while robot_copy.alive and not done and np.max(count) < 3:
            # Move
            state_, reward = model_free.simulation(robot_copy, action)
            policy_ = model_free.policy[:, state_[0], state_[1]]
            action_ = choice(model_free.directions, p=policy_)
            count[state[0]][state[1]] += 1

            # update Q table
            model_free.update_Qvalue(action, state, state_, reward, alpha, gamma, True, action_)
            #print(model_free.Qvalue_table)
            # update policy
            model_free.update_policy(epsilon, state)
            action = action_
            state = state_

            # if the cleanliness percentage is 100
            clean = (robot_copy.grid.cells == 0).sum()
            dirty = (robot_copy.grid.cells >= 1).sum()
            cleanliness = clean / (clean + dirty)
            # print(cleanliness)
            if cleanliness == 1:
                done = True
        episodes -= 1
        #print(model_free.Qvalue_table)
    return model_free.policy


def robot_epoch(robot):
    model_free = TD(robot)
    # alpha gamma epsilon episode
    optimal_policy = sarsa(model_free, 0.1, 0.4, 0.2, 500)# 0.1, 1, 0.35, 500 for death with 100 runs
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    for index in range(0, 4):
        if index in indices:
            probability.append(1 / len(indices))
        else:
            probability.append(0)
    direction = choice(model_free.directions, p=probability)
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()

def robot_epoch_(robot, lr, gamma, epsilon):
    model_free = TD(robot)
    optimal_policy = sarsa(model_free, lr, gamma, epsilon, 500)# 0.1, 1, 0.35, 500 for death with 100 runs
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    for index in range(0, 4):
        if index in indices:
            probability.append(1 / len(indices))
        else:
            probability.append(0)
    direction = choice(model_free.directions, p=probability)
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()