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

def get_current_rewards(cells):
    """
    Get the reward matrix based on grid's current circumstances(each tile's label) and robot's history.
    :param cells: cells attribute of robot.grid, a matrix record the label of each tile
    :param transformation: a punishment matrix, where each element is the punishment of each tile
    :returns combined_reward: a reward matrix
    """
    reward = copy.deepcopy(cells)
    # label < -2: this tile has a robot with different direction inside it. We set it to 0, meaning it is already clean.
    reward[reward < -2] = 0
    # label -2: this tile is an obstacle, we think they have the same function of wall tiles, so we reset as -1
    reward[reward == -2] = -1
    # label 3: death tile, give -3 to avoid robot reach it.
    reward[reward == 3] = -3
    max_value = np.max(reward)
    if max_value < 1:
        # After all the tiles have been cleared
        # the robot must visit the death tile to terminate, so give it a high value 3
        reward[reward == -3] = 3
    return reward

def simulation(robot, action, frequency):
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
    #     reward = - frequency[robot.pos[0]][robot.pos[1]]*0.01
    return robot.pos, reward

def Q_learning(robot, alpha, gamma, epsilon, episodes):
    # initialize parameters
    n_cols = robot.grid.n_rows
    n_rows = robot.grid.n_cols
    policy = init_policy(n_rows, n_cols)
    Qvalue_table = init_Qvalue_table(n_rows, n_cols)
    directions = ['n', 'e', 's', 'w']
    direction_index_map = {'n':0, 'e':1, 's':2, 'w':3}
    frequency = np.zeros((n_rows, n_cols))
    print("overall start")
    while episodes:
        robot_copy = copy.deepcopy(robot)
        while robot_copy.alive and np.max(robot_copy.grid.cells) > 0 and np.max(frequency) < 20:
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
            next_state, reward = simulation(robot_copy, action, frequency)
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

        episodes -= 1
    return policy

def robot_epoch(robot):
    directions = ['n', 'e', 's', 'w']
    optimal_policy = Q_learning(robot, 0.1, 1, 0.2, 500)
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    direction = choice(directions, p=policy_of_current_state)
    # direction = directions[np.argmax(policy_of_current_state)]
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()
