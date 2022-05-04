import numpy as np
import copy
import random

# initialization function
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

def init_values(n_rows, n_cols):
    return np.full((n_rows, n_cols), 0)

# policy iteration algorithm
def policy_evaluation(dirs, rewards, values, policy):
    value = np.full(rewards.shape, 0)
    return value

def policy_improvement(dirs, rewards, values, policy):
    policy_stable = True
    n_rows = rewards.shape[0]
    n_cols = rewards.shape[1]
    # action_values = init_action_values(n_rows, n_cols)
    for i in range(0, n_rows):
        for j in range(0, n_cols):
            old_policy = policy[i][j]
            action_values = {'n': 0, 'e': 0, 's': 0, 'w': 0}
            # calculate Q(s,a)
            for action in dirs:
                next_i = i + dirs[action][0]
                if next_i > n_rows - 1:
                    next_i = n_rows - 1
                if next_i < 0:
                    next_i = 0
                next_j = j + dirs[action][1]
                if next_j > n_cols - 1:
                    next_j = n_cols - 1
                if next_j < 0:
                    next_j = 0
                action_values[action] = rewards[next_i][next_j] + values[next_i][next_j]
            # get the highest Q(s,a) for every s
            max_action_value = [key for m in [max(action_values.values())] for key, val in action_values.items() if val == m]
            # change the policy of state i, j
            new_policy = {'n': 0, 'e': 0, 's': 0, 'w': 0}
            probability = 1/len(max_action_value)
            for action in max_action_value:
                new_policy[action] = probability
            if not new_policy == old_policy:
                policy_stable = False
                policy[i][j] = new_policy
    return policy, policy_stable

def policy_iteration(robot):
    # initialize parameters
    n_rows = robot.grid.n_rows
    n_cols = robot.grid.n_cols
    policy = init_policy(n_rows, n_cols)
    rewards = get_current_rewards(robot.grid.cells)
    values = init_values(n_rows, n_cols)
    dirs = robot.dirs
    policy_stable = False
    # do iteration
    while not policy_stable:
        values = policy_evaluation(dirs, rewards, values, policy)
        policy, policy_stable = policy_improvement(dirs, rewards, values, policy)
    return policy


# parameter initialization
# global optimal_policy
# global find_optimal_policy
# find_optimal_policy = True

def robot_epoch(robot):
    # get current state's optimal policy
    optimal_policy = policy_iteration(robot)
    policy_of_current_pos = optimal_policy[robot.pos[0]][robot.pos[1]]
    direction = random.choices(list(policy_of_current_pos.keys()), weights=policy_of_current_pos.values(), k=1)[0]
    while direction != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()
