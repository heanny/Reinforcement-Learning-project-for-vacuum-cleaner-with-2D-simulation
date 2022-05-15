import numpy as np
from numpy.random import choice
import copy
from model_free import ModelFree

def Q_learning(model_free, alpha, gamma, epsilon, episodes):
    # initialize parameters
    # n_cols = robot.grid.n_rows
    # n_rows = robot.grid.n_cols
    # policy = init_policy(n_rows, n_cols)
    # Qvalue_table = init_Qvalue_table(n_rows, n_cols)
    # directions = ['n', 'e', 's', 'w']
    # direction_index_map = {'n':0, 'e':1, 's':2, 'w':3}
    frequency = np.zeros((model_free.n_rows, model_free.n_cols))
    # policy_not_stable = True
    while episodes:
        robot_copy = copy.deepcopy(model_free.robot)
        not_finished = True
        # and np.max(frequency) < 20
        while robot_copy.alive and not_finished and np.max(frequency) < 20:
            print("+++++++++++++++++++++++ start +++++++++++++++++++++++++++++++")
            print(robot_copy.alive, np.max(robot_copy.grid.cells))
            # current state
            state = robot_copy.pos
            print("current state", state)
            i = state[0]
            j = state[1]
            frequency[i][j] += 1
            # use policy to choose action given state
            policy_of_current_state = model_free.policy[:, i, j]
            action = choice(model_free.directions, p=policy_of_current_state)

            # simulate and get s' and r
            print("start simulation")
            next_state, reward = model_free.simulation(robot_copy, action)
            print("end simulation")
            print(next_state, reward)

            # update Qvalue table
            model_free.update_Qvalue(action, state, next_state, reward, alpha, gamma, False, None)

            # update epsilon-greedy policy
            model_free.update_policy(epsilon, state)

            # if (old_policy == policy[:, i, j]).all():
            #     policy_not_stable = False
            clean = (robot_copy.grid.cells == 0).sum()
            dirty = (robot_copy.grid.cells >= 1).sum()
            if clean/(clean+dirty) == 1:
                not_finished = False
            print("+++++++++++++++++++++++ end +++++++++++++++++++++++++++++++")

        episodes -= 1
    return model_free.policy

def robot_epoch(robot):
    model_free = ModelFree(robot)
    optimal_policy = Q_learning(model_free, 0.1, 1, 0.2, 500)
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    for index in range(0, 4):
        if index in indices:
            probability.append(1/len(indices))
        else:
            probability.append(0)
    direction = choice(model_free.directions, p=probability)
    print(direction)
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()