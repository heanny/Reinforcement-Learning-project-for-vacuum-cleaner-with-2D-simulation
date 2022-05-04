import numpy as np
import copy
import random
# sys.path.append("..")
# from app import robots, grid

# n_cols = robots.grid.n_rows
# n_rows = robots.grid.n_cols
# history = np.zeros(n_cols, n_rows)

# initialization function
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
    # rounded=np.round_(transformation,0)
    # print(rounded)
    combined_reward=reward+transformation
    # combined_reward= combined_reward.astype(int)
    # print(combined_reward)
    return combined_reward

def init_values(n_rows, n_cols):
    return np.full((n_rows, n_cols), 0)

# policy iteration algorithm
def policy_evaluation(dirs, rewards, values, policy,gamma=1.0, theta=1, max_iterations=1e9):
    evaluation_iterations = 1
    rows = rewards.shape[0]
    cols = rewards.shape[1]
    # Repeat until change in value is below the threshold
    for l in range(int(max_iterations)):
        # Initialize a change of value function as zero
        delta = 0
        # Iterate though each state
        for i in range(rows):
            for j in range(cols):
                # Initial a new value of current state
                v = 0
                # Look at the possible next actions
                # dir_list=list(policy[i][j])
                for key in policy[i][j].keys():
                    # Look at the possible next states for each action
                    action_prob = policy[i][j].get(key)
                    row = i+dirs.get(key)[1]
                    col = j+dirs.get(key)[0]
                    if not (row < int(rows) and col < int(col)):
                        row = i
                        col = j
                    v += action_prob * (rewards[i][j] + gamma * values[row][col])
                # Calculate the absolute change of value function
                delta = max(delta, np.abs(values[i][j] - v))
                # Update value function
                # print(rewards)
                # print(v)
                values[i][j] = int(v)
        evaluation_iterations += 1
        # Terminate if value change is insignificant
        if delta < theta:
            print(f'Policy evaluated in {evaluation_iterations} iterations.')
            return values

def policy_improvement(dirs, rewards, values, policy,gamma=1.0):
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
                action_values[action] = rewards[next_i][next_j] + gamma*values[next_i][next_j]
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

def policy_iteration(robot,transformation):
    # initialize parameters
    n_cols = robot.grid.n_rows
    n_rows = robot.grid.n_cols
    policy = init_policy(n_rows, n_cols)
    rewards = get_current_rewards(robot.grid.cells,transformation)
    values = init_values(n_rows, n_cols)
    dirs = robot.dirs
    policy_stable = False
    # do iteration
    while not policy_stable:
        values = policy_evaluation(dirs, rewards, values, policy, gamma=1, theta=20, max_iterations=1e9)
        policy, policy_stable = policy_improvement(dirs, rewards, values, policy, gamma=1)
    return policy


# parameter initialization
# global optimal_policy
# global find_optimal_policy
# find_optimal_policy = True

def robot_epoch(robot):
    if not any(robot.history):
        print("no history")
        n_cols = robot.grid.n_rows
        n_rows = robot.grid.n_cols
        global history
        history = np.full((n_rows, n_cols),0.0)
        # print(history)
    e = np.finfo(float).eps
    # print(history)
    history_updated=np.where(history<99,history,99)
    transformation = np.where(history==0, history, -0.01*history)
    # (e**(-history+1))-1
    # print(transformation)
    # get current state's optimal policy
    optimal_policy = policy_iteration(robot,transformation)
    policy_of_current_pos = optimal_policy[robot.pos[0]][robot.pos[1]]
    direction = random.choices(list(policy_of_current_pos.keys()), weights=policy_of_current_pos.values(), k=1)[0]
    while direction != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')

    # Move:
    position=robot.pos
    history[position[0]][position[1]] += 1
    # print(history)
    robot.move()