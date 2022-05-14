import numpy as np
import copy
import random

# initialization function
def init_policy(n_rows, n_cols):
    """
    Initialize the policy matrix, where each element is a dictionary that shows
    the probability of moving in a certain direction in a given state.
    We initialize each direction with a probability of 1 in 4.
    :param n_rows: number of rows in the grid
    :param n_cols: number of columns in the grid
    :returns policy: the policy matrix
    """
    d = {'n': 0.25, 'e': 0.25, 's': 0.25, 'w': 0.25}
    policy = np.full((n_rows, n_cols), d)
    return policy

def init_Qvalue_table(n_rows, n_cols):
    """
    Initialize the Q-value table, where each element is a dictionary that shows
    the value of moving in a certain direction in a given state.
    We initialize every state-action pair as 0.
    :param n_rows: number of rows in the grid
    :param n_cols: number of columns in the grid
    :returns policy: the Q-value table
    """
    d = {'n': 0, 'e': 0, 's': 0, 'w': 0}
    Qvalue_table = np.full((n_rows, n_cols), d)
    return Qvalue_table

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

def simulation(robot, action):
    # get reward of action
    coordinate = robot.dirs[action]
    next_i = robot.pos[0] + coordinate[0]
    next_j = robot.pos[1] + coordinate[1]
    reward = get_current_rewards(robot.grid.cells)[next_i][next_j]
    # take action
    while action != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    robot.move()
    # return the new state s' and reward
    return robot.pos, reward

def Q_learning(robot, alpha=0.1, gamma=1, epsilon=0.2, episodes=50):
    # initialize parameters
    n_cols = robot.grid.n_rows
    n_rows = robot.grid.n_cols
    policy = init_policy(n_rows, n_cols)
    Qvalue_table = init_Qvalue_table(n_rows, n_cols)
    while episodes:
        robot_copy = copy.deepcopy(robot)
        while robot_copy.alive:
            # current state
            state = robot.pos
            # use policy to choose action given state
            policy_of_current_state = policy[robot.pos[0]][robot.pos[1]]
            action = random.choices(list(policy_of_current_state.keys()), weights=policy_of_current_state.values(), k=1)[0]
            # simulate and get s' and r
            next_state, reward = simulation(robot_copy, action)

            # update Qvalue table
            old_Qvalue = Qvalue_table[state[0]][state[1]][action]  # get Q(s,a)
            next_state_Qvalues = Qvalue_table[next_state[0]][next_state[1]]  # get the dictionary of s' Qvalue
            next_state_max_Qvalue = next_state_Qvalues[max(next_state_Qvalues)]  # get max Qvalue of s'
            Qvalue_table[state[0]][state[1]][action] = old_Qvalue + alpha * (reward + gamma * next_state_max_Qvalue + old_Qvalue)

            # update epsilon-greedy policy
            Qvalue_table = Qvalue_table[state[0]][state[1]]  # get current state's Qvalue dictionary
            max_action_value = [key for m in [max(Qvalue_table.values())] for key, val in Qvalue_table.items() if val == m]  # get the highest Q(s,a) for s, there could be more than 1 highest Q(s,a)
            smallest_probability = epsilon/4  # smallest_probability for maintaining exploration
            new_policy = {'n': smallest_probability, 'e': smallest_probability, 's': smallest_probability, 'w': smallest_probability}
            greedy_probability = (1-epsilon)/len(max_action_value) + epsilon/4
            for action in max_action_value:
                new_policy[action] = greedy_probability
    return policy

def robot_epoch(robot):
    optimal_policy = Q_learning(robot)
    policy_of_current_pos = optimal_policy[robot.pos[0]][robot.pos[1]]
    direction = random.choices(list(policy_of_current_pos.keys()), weights=policy_of_current_pos.values(), k=1)[0]
    while direction != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()