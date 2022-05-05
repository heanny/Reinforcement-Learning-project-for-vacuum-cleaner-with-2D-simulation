import numpy as np
import random
import copy


def init_values(n_rows, n_cols):
    return np.full((n_rows, n_cols), 0)

def init_policy(n_rows, n_cols):
    d = {'n': 0.25, 'e': 0.25, 's': 0.25, 'w': 0.25}
    policy = np.full((n_rows, n_cols), d)
    return policy

def get_current_rewards(cells,transformation):
    reward = copy.deepcopy(cells)
    reward[reward < -2] = 0
    reward[reward == -2] = -1
    reward[reward == 3] = -3
    max_value = np.max(reward)
    if max_value < 1:
        reward[reward == -3] = 3
    return reward+transformation


def Value_iteration(n,gamma,robot,transformation):
    dirs = robot.dirs
    rewards = get_current_rewards(robot.grid.cells,transformation)
    # print(rewards)

    threshold = 5

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
        update_value_table = init_values(grid_n_rows, grid_n_cols)
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
                    action_value[action] = rewards[next_i][next_j] + gamma * value_table[next_i][next_j]

                update_value_table[i][j] = action_value[max(action_value)]
                max_action_value = [key for m in [max(action_value.values())] for key, val in action_value.items() if
                                    val == m]
                new_policy = {'n': 0, 'e': 0, 's': 0, 'w': 0}
                probability = 1 / len(max_action_value)
                for action in max_action_value:
                    new_policy[action] = probability
                policy[i][j] = new_policy
        if np.max((np.fabs(update_value_table - value_table))) < threshold:
            break
        else:
            value_table = update_value_table

    return policy


def robot_epoch(robot):
    if not any(robot.history):
        # print("no history")
        n_cols = robot.grid.n_rows
        n_rows = robot.grid.n_cols
        global history
        history = np.full((n_rows, n_cols),0.0)
        # print(history)
    # e = np.finfo(float).eps
    history = np.where(history < 99, history, 99)
    transformation = np.where(history==0, history, -0.01*history)
    # print(transformation)

    # get current state's optimal policy
    optimal_policy = Value_iteration(1000,1,robot,np.round(transformation,5))
    policy_of_current_pos = optimal_policy[robot.pos[0]][robot.pos[1]]
    direction = random.choices(list(policy_of_current_pos.keys()), weights=policy_of_current_pos.values(), k=1)[0]
    while direction != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')

    # next_pos = np.array(robot.pos) + np.array(robot.dirs[direction])
    # #print(next_pos)
    # history[next_pos[0]][next_pos[1]] += 1
    # print(history)
    # e = np.finfo(float).eps
    # history = np.where(history==0, history, e**(-history+1)-1)
    # print(history)

    # Move:
    position=robot.pos
    history[position[0]][position[1]] += 1
    # print(history)
    robot.move()