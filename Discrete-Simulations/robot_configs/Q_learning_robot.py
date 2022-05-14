import numpy as np
import copy
import random
from numpy.random import choice

# initialization function
def init_policy(n_rows, n_cols):
    """
    Initialize the policy matrix, where each element is a dictionary that shows
    the probability of moving in a certain direction in a given state.
    We initialize each direction with a probability of 1 in 4.
    :param n_rows: number of rows in the grid
    :param n_cols: number of columns in the grid
    :returns policy: the policy 3D matrix
    """
    initial_pobability = 0.25
    policy = np.full((4, n_rows, n_cols), initial_pobability)
    return policy

def init_Qvalue_table(n_rows, n_cols):
    """
    Initialize the Q-value table, where each element is a dictionary that shows
    the value of moving in a certain direction in a given state.
    We initialize every state-action pair as 0.
    :param n_rows: number of rows in the grid
    :param n_cols: number of columns in the grid
    :returns policy: the 3D Q-value matrix
    """
    return np.zeros((4,n_rows, n_cols))

def simulation(robot, action, transformation):
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
    if reward == 0:
        reward = transformation[robot.pos[0]][robot.pos[1]]
    return robot.pos, reward

def Q_learning(robot, transformation, alpha, gamma, epsilon, episodes):
    # initialize parameters
    n_cols = robot.grid.n_rows
    n_rows = robot.grid.n_cols
    policy = init_policy(n_rows, n_cols)
    Qvalue_table = init_Qvalue_table(n_rows, n_cols)
    directions = ['n', 'e', 's', 'w']
    direction_index_map = {'n':0, 'e':1, 's':2, 'w':3}
    frequency = np.zeros((n_rows, n_cols))
    while episodes:
        robot_copy = copy.deepcopy(robot)
        while robot_copy.alive and np.max(robot_copy.grid.cells) > 0 and np.max(frequency) < 20:
            print("+++++++++++++++++++++++ start +++++++++++++++++++++++++++++++")
            print(robot_copy.alive, np.max(robot_copy.grid.cells))
            # current state
            state = robot_copy.pos
            print("current state", state)
            i = state[0]
            j = state[1]
            frequency[i][j] += 1
            # use policy to choose action given state
            policy_of_current_state = policy[:, i, j]
            action = choice(directions, p=policy_of_current_state)

            # simulate and get s' and r
            print("start simulation")
            next_state, reward = simulation(robot_copy, action, transformation)
            print("end simulation")
            print(next_state, reward)

            # update Qvalue table
            action_index = direction_index_map[action]
            old_Qvalue = Qvalue_table[action_index, i, j]  # get Q(s,a)
            print("old Qvalue:", old_Qvalue)
            next_state_Qvalues = Qvalue_table[:, next_state[0], next_state[1]]  # get all the Q(s',a)
            next_state_max_Qvalue = max(next_state_Qvalues)  # get max Qvalue of s'
            Qvalue_table[action_index, i, j] = old_Qvalue + alpha * (reward + gamma * next_state_max_Qvalue + old_Qvalue)
            print("new Qvalue:", Qvalue_table[action_index, i, j])

            # update epsilon-greedy policy
            print("old policy:", policy[:, i, j])
            Qvalues = Qvalue_table[:, i, j]  # get current state all Qvalues
            max_Qvalue = max(Qvalues)  # get the highest Q(s,a) for s, there could be more than 1 highest Q(s,a)
            indices = [index for index, value in enumerate(Qvalues) if value == max_Qvalue] # find the indices of all max value
            smallest_probability = epsilon/4  # smallest_probability for maintaining exploration
            greedy_probability = (1-epsilon)/len(indices) + epsilon/4
            for index in range(0, 4):
                if index in indices:
                    policy[index, i, j] = greedy_probability
                else:
                    policy[index, i, j] = smallest_probability
            print("new policy:", policy[:, i, j])
            print("+++++++++++++++++++++++ end +++++++++++++++++++++++++++++++")
        episodes -= 1
    return policy

def robot_epoch(robot):
    directions = ['n', 'e', 's', 'w']

    if not any(robot.history):
        n_cols = robot.grid.n_rows
        n_rows = robot.grid.n_cols
        global history
        history = np.full((n_rows, n_cols), 0.0)

    history = np.where(history < 99, history, 99)
    transformation = np.where(history == 0, history, -0.01 * history)  # the range of each element is (-1,0]

    optimal_policy = Q_learning(robot, transformation, 0.1, 1, 0.4, 500)
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    for index in range(0, 4):
        if index in indices:
            probability.append(1/len(indices))
        else:
            probability.append(0)
    direction = choice(directions, p=probability)
    print(direction)
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()
