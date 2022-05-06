import numpy as np
import random
import copy


def init_values(n_rows, n_cols):
    """
    Initialize the value matrix, where each element is a value used to evaluate a state.
    We initialize value of each state with 0.
    :param n_rows: number of rows in the grid
    :param n_cols: number of columns in the grid
    :returns policy: the value matrix
    """

    return np.full((n_rows, n_cols), 0)

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

def get_current_rewards(cells,transformation):
    """
    Get the reward matrix based on grid's current circumstances(each tile's label) and robot's history.
    :param cells: cells attribute of robot.grid, a matrix record the label of each tile
    :param transformation: a punishment matrix, where each element is the punishment of each tile
    :returns combined_reward: a reward matrix
    """
    reward = copy.deepcopy(cells)
    reward[reward < -2] = 0
    reward[reward == -2] = -1
    reward[reward == 3] = -3
    max_value = np.max(reward)
    if max_value < 1:
        reward[reward == -3] = 3
    return reward+transformation


def Value_iteration(n,gamma,robot,transformation):
    """
    When the value function converges, end the iteration to return the best policy.
    :param n: the number of iterations
    :param gamma: discount factor
    :param robot: robot
    :param transformation: a punishment matrix, where each element is the punishment of each tile
    """
    # get directions of robot
    dirs = robot.dirs
    # get reward matrix
    rewards = get_current_rewards(robot.grid.cells,transformation)

    # set the value of theta
    theta = 5

    rewards_n_rows = rewards.shape[0]
    rewards_n_cols = rewards.shape[1]

    grid_n_cols = robot.grid.n_rows
    grid_n_rows = robot.grid.n_cols

    value_table = init_values(grid_n_rows, grid_n_cols)
    policy = init_policy(grid_n_rows, grid_n_cols)

    # start iteration
    for k in range(n):
        # create null state value table
        update_value_table = init_values(grid_n_rows, grid_n_cols)
        # traverse all states
        for i in range(0, rewards_n_rows):
            for j in range(0, rewards_n_cols):
                action_value = {'n': 0, 'e': 0, 's': 0, 'w': 0}
                # traverse all actions
                for action in dirs:
                    # Check if the boundary is reached
                    next_i = i + dirs[action][0]
                    if next_i > rewards_n_rows - 1:
                        next_i = rewards_n_rows - 1
                    if next_i < 0:
                        next_i = 0
                    next_j = j + dirs[action][1]
                    if next_j > rewards_n_cols - 1:
                        next_j = rewards_n_cols - 1
                    if next_j < 0:
                        next_j = 0
                    # iteration equation
                    action_value[action] = rewards[next_i][next_j] + gamma * value_table[next_i][next_j]

                # update the value table by the max value of action values
                update_value_table[i][j] = action_value[max(action_value)]
                max_action_value = [key for m in [max(action_value.values())] for key, val in action_value.items() if
                                    val == m]
                new_policy = {'n': 0, 'e': 0, 's': 0, 'w': 0}
                probability = 1 / len(max_action_value)
                for action in max_action_value:
                    new_policy[action] = probability
                policy[i][j] = new_policy

        # Check for convergence
        if np.max((np.fabs(update_value_table - value_table))) < theta:
            break
        else:
            value_table = update_value_table

    return policy


def robot_epoch(robot):
    if not any(robot.history):

        n_cols = robot.grid.n_rows
        n_rows = robot.grid.n_cols
        global history
        history = np.full((n_rows, n_cols),0.0)

    history = np.where(history < 99, history, 99)
    transformation = np.where(history==0, history, -0.01*history)

    # get current state's optimal policy
    optimal_policy = Value_iteration(1000,1,robot,np.round(transformation,5))
    policy_of_current_pos = optimal_policy[robot.pos[0]][robot.pos[1]]
    direction = random.choices(list(policy_of_current_pos.keys()), weights=policy_of_current_pos.values(), k=1)[0]
    while direction != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')

    # Move:
    position = robot.pos
    history[position[0]][position[1]] += 1
    # print(history)
    robot.move()