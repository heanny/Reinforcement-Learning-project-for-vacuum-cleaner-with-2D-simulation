import numpy as np
from numpy.random import choice
import copy
from td import TD

def Q_learning(model_free, alpha, gamma, epsilon, episodes):
    """
    This function implement the TD off-policy control(Q_learning)
    :param model_free: temporal difference class TD from td.py
    :param alpha: the learning rate
    :param gamma: discounted factor
    :param epsilon: epsilon-greedy factor
    :returns policy: the optimal policy matrix
    """
    while episodes:
        robot_copy = copy.deepcopy(model_free.robot)
        # accumulate the state's visit time in an episodes
        frequency = np.zeros((model_free.n_rows, model_free.n_cols))
        # is the episodes terminate or not
        not_finished = True

        # enter an episode
        while robot_copy.alive and not_finished and np.max(frequency) < 3:
            # current state
            state = robot_copy.pos
            frequency[state[0]][state[1]] += 1

            # use epsilon-greedy policy to simulate the next action in this episode
            policy_of_current_state = model_free.policy[:, state[0], state[1]]
            action = choice(model_free.directions, p=policy_of_current_state)
            # get the r and s' after action
            next_state, reward = model_free.simulation(robot_copy, action)

            # update Q value of current state
            model_free.update_Qvalue(action, state, next_state, reward, alpha, gamma, False, None)

            # update epsilon-greedy policy
            model_free.update_policy(epsilon, state)

            # judge the cleanness to see if episode is finished or not
            clean = (robot_copy.grid.cells == 0).sum()
            dirty = (robot_copy.grid.cells >= 1).sum()
            if clean/(clean+dirty) == 1:
                not_finished = False

        episodes -= 1
    return model_free.policy

def robot_epoch(robot):
    # initial TD class
    model_free = TD(robot)
    # use parameter alpha, gamma, epsilon and episodes to start Q-learning algorithm and get optimal policy
    # the optimal policy is the policy we get from Q-learning algorithm
    optimal_policy = Q_learning(model_free, alpha=0.3, gamma=0.2, epsilon=0.4, episodes=500)# this is the optimal parameter settings for app.py
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    for index in range(0, 4):
        if index in indices:
            probability.append(1/len(indices))
        else:
            probability.append(0)
    # based on optimal policy choose an action
    direction = choice(model_free.directions, p=probability)
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()

# The robot epoch function for multiple processing
def robot_epoch_(robot, lr, gamma, epsilon):
    # initial TD class
    model_free = TD(robot)
    # use parameter alpha, gamma, epsilon and episodes to start Q-learning algorithm and get optimal policy
    optimal_policy = Q_learning(model_free, lr, gamma, epsilon, 500)# '500': here to set the smaller iterations to get the heatmap.
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    for index in range(0, 4):
        if index in indices:
            probability.append(1/len(indices))
        else:
            probability.append(0)
    # based on optimal policy choose an action
    direction = choice(model_free.directions, p=probability)
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()
