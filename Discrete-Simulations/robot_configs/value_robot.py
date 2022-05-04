import numpy as np
import random
import copy


def init_values(n_rows, n_cols):
    return np.full((n_rows, n_cols), 0)

def init_policy(n_rows, n_cols):
    d = {'n': 0.25, 'e': 0.25, 's': 0.25, 'w': 0.25}
    policy = np.full((n_rows, n_cols), d)
    return policy

def get_current_rewards(cells):
    reward = copy.deepcopy(cells)
    reward[reward < -2] = 0
    reward[reward == -2] = -1
    reward[reward == 3] = -3
    max_value = np.max(reward)
    if max_value < 1:
        reward[reward == -3] = 3
    return reward


def Value_iteration(n,gamma,robot):
    dirs = robot.dirs
    rewards = get_current_rewards(robot.grid.cells)
    threshold = 1e-30

    rewards_n_rows = rewards.shape[0]
    rewards_n_cols = rewards.shape[1]
    # grid_n_rows = robot.grid.n_rows
    # grid_n_cols = robot.grid.n_cols
    grid_n_cols = robot.grid.n_rows
    grid_n_rows = robot.grid.n_cols

    # value_table = init_values(grid_n_rows, grid_n_cols)
    # policy = init_policy(grid_n_rows, grid_n_cols)
    value_table = init_values(grid_n_rows, grid_n_cols)
    policy = init_policy(grid_n_rows, grid_n_cols)

    #start iteration
    for k in range(n):
        # create null state value table
        update_value_table = np.copy(value_table)
        #traverse all states
        for i in range(0, rewards_n_rows):
            for j in range(0, rewards_n_cols):
                action_value = {'n': 0, 'e': 0, 's': 0, 'w': 0}
                # traverse all actions
                for action in dirs:
                    # print(f"dirs:{action}")
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


                    action_value[action] = rewards[next_i][next_j] + gamma * 1 * update_value_table[next_i][next_j]

                max_action_value = [key for m in [max(action_value.values())] for key, val in action_value.items() if
                                    val == m]
                # print(max_action_value)
                new_policy = {'n': 0, 'e': 0, 's': 0, 'w': 0}
                probability = 1 / len(max_action_value)
                for action in max_action_value:
                    new_policy[action] = probability
                policy[i][j] = new_policy
        if np.sum((np.fabs(update_value_table - value_table))) <= threshold:
            break

    return policy


def robot_epoch(robot):
    # get current state's optimal policy
    optimal_policy = Value_iteration(1000,1,robot)
    policy_of_current_pos = optimal_policy[robot.pos[0]][robot.pos[1]]
    direction = random.choices(list(policy_of_current_pos.keys()), weights=policy_of_current_pos.values(), k=1)[0]
    while direction != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()