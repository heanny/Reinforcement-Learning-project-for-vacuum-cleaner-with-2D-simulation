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

def get_current_rewards(cells, transformation):
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
    # upon a current reward matrix, we add a punishment matrix generated from robot history.
    combined_reward=reward+transformation
    return combined_reward

def init_values(n_rows, n_cols):
    """
    Initialize the value matrix, where each element is a value used to evaluate a state.
    We initialize value of each state with 0.
    :param n_rows: number of rows in the grid
    :param n_cols: number of columns in the grid
    :returns policy: the value matrix
    """
    return np.full((n_rows, n_cols), 0)

# policy iteration algorithm
def policy_evaluation(dirs, rewards, values, policy,gamma=1.0, theta=1, max_iterations=1e9):
    """
    Initialize the value matrix, where each element is a value used to evaluate a state.
    We initialize value of each state with 0.
    :param dirs: the possible directions of robot
    :param rewards: reward matrix
    :param values: value matrix
    :param policy: policy matrix
    :param gamma: gamma parameter to multiply with the value of new state
    :param theta: the threshold for convergence
    :param max_iterations: the maximum number of iterations
    :returns values: the converged value matrix, updated by the policy
    """
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
                for key in policy[i][j].keys():
                    # find the next state for each action
                    action_prob = policy[i][j].get(key)
                    row = i+dirs.get(key)[0]
                    col = j+dirs.get(key)[1]
                    # if the next state is out of bound, use the current state
                    if not (row < int(rows) and col < int(col)):
                        row = i
                        col = j
                    v += action_prob * (rewards[i][j] + gamma * values[row][col])
                # Calculate the absolute change of value function
                delta = max(delta, np.abs(values[i][j] - v))
                # Update value function
                values[i][j] = int(v)
        evaluation_iterations += 1
        # Terminate if value change is insignificant
        if delta < theta:
            print(f'Policy evaluated in {evaluation_iterations} iterations.')
            return values

def policy_improvement(dirs, rewards, values, policy, gamma=1.0):
    """
    Initialize the value matrix, where each element is a value used to evaluate a state.
    We initialize value of each state with 0.
    :param dirs: robot's dirs attribute
    :param rewards: reward matrix
    :param values: value matrix
    :param policy: policy matrix
    :param gamma: gamma parameter to multiply with V(s')
    :returns policy: the new policy matrix generated by current V
    :returns policy_stable: a boolean indicates whether the policy has changed
    """
    policy_stable = True
    n_rows = rewards.shape[0]
    n_cols = rewards.shape[1]
    # iterate over each state
    for i in range(0, n_rows):
        for j in range(0, n_cols):
            # store the old policy
            old_policy = policy[i][j]
            action_values = {'n': 0, 'e': 0, 's': 0, 'w': 0}
            # iterate over each action in a given state, calculate Q(s,a)
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
            # get the highest Q(s,a) for every s, there could be more than 1 highest Q(s,a)
            max_action_value = [key for m in [max(action_values.values())] for key, val in action_values.items() if val == m]
            # change the policy of state i, j
            new_policy = {'n': 0, 'e': 0, 's': 0, 'w': 0}
            probability = 1/len(max_action_value)
            for action in max_action_value:
                new_policy[action] = probability
            # compare to old policy to check if the policy change or not
            if not new_policy == old_policy:
                policy_stable = False
                policy[i][j] = new_policy
    return policy, policy_stable

def policy_iteration(robot,transformation):
    """
    the main body of policy iteration algorithm
    :param robot: robot
    :param transformation: a punishment matrix, where each element is the punishment of each tile.
     it will be use to calculate reward matrix.
    :returns policy: the optimal policy
    """
    # initialize parameters
    n_cols = robot.grid.n_rows
    n_rows = robot.grid.n_cols
    policy = init_policy(n_rows, n_cols)
    rewards = get_current_rewards(robot.grid.cells,transformation)
    values = init_values(n_rows, n_cols)
    dirs = robot.dirs
    policy_stable = False
    # start iteration
    while not policy_stable:
        values = policy_evaluation(dirs, rewards, values, policy, gamma=1, theta=20, max_iterations=1e9)
        policy, policy_stable = policy_improvement(dirs, rewards, values, policy, gamma=1)
    return policy


def robot_epoch(robot):
    # initialize global variable "history" when starting the robot, to add the cleaned tiles in a matrix
    if not any(robot.history):
        n_cols = robot.grid.n_rows
        n_rows = robot.grid.n_cols
        global history
        history = np.full((n_rows, n_cols),0.0)
    
    history = np.where(history<99,history,99)
    transformation = np.where(history==0, history, -0.01*history) # the range of each element is (-1,0]
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
    robot.move()