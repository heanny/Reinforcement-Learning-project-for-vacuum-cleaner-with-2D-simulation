import numpy as np
from numpy.random import choice
import copy
from td import TD

def Q_learning(model_free, alpha, gamma, epsilon, episodes):
    while episodes:
        robot_copy = copy.deepcopy(model_free.robot)
        frequency = np.zeros((model_free.n_rows, model_free.n_cols))
        not_finished = True
        # enter an episode
        while robot_copy.alive and not_finished and np.max(frequency) < 3:
            # current state
            state = robot_copy.pos
            frequency[state[0]][state[1]] += 1

            # simulate and get s' and r
            # use policy to choose action given state
            policy_of_current_state = model_free.policy[:, state[0], state[1]]
            action = choice(model_free.directions, p=policy_of_current_state)
            next_state, reward = model_free.simulation(robot_copy, action)

            # update Qvalue table
            model_free.update_Qvalue(action, state, next_state, reward, alpha, gamma, False, None)

            # update epsilon-greedy policy
            model_free.update_policy(epsilon, state)

            # judge the cleanness
            clean = (robot_copy.grid.cells == 0).sum()
            dirty = (robot_copy.grid.cells >= 1).sum()
            if clean/(clean+dirty) == 1:
                not_finished = False

        episodes -= 1
    return model_free.policy

def robot_epoch(robot):
    model_free = TD(robot)
    optimal_policy = Q_learning(model_free, 0.1, 1, 0.3, 300)
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    for index in range(0, 4):
        if index in indices:
            probability.append(1/len(indices))
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
    optimal_policy = Q_learning(model_free, lr, gamma, epsilon, 500)
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    for index in range(0, 4):
        if index in indices:
            probability.append(1/len(indices))
        else:
            probability.append(0)
    direction = choice(model_free.directions, p=probability)
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()