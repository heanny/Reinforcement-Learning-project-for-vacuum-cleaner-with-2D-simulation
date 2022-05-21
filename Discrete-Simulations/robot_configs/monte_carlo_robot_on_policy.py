import random
from collections import defaultdict
from copy import deepcopy
import numpy as np
from mc import MC

def on_policy_mc_control(MC): 
    """
    Implement the on policy Monte Carlo control
    :param MC: the Monte Carlo class (from MC.py)
    :returns policy: the optimal policy
    """
    # initialization
    policy = MC.policy
    epsilon = MC.epsilon
    # repeat for l iterations:
    for l in range(int(MC.max_iteration)):
        # generate an episode
        episode = MC.generate_episode(policy)
        # Update Q table for each (s,a) in episode
        MC.Q_table(episode)
        Q_tmp = MC.Q
        for item in episode:  # item: (state,action,reward)
            state = item[0]
            # List of Q values of actions correspond to this state
            A = Q_tmp[:, state[0], state[1]]
            # Find the action with the highest Q values
            best_action = np.argmax(A)
            # In case there are more than one best actions: find all the best actions
            best_action_indices = np.where(A == Q_tmp[best_action, state[0], state[1]])
            # update policy with epsilon-greedy
            for action in range(4):
                if action in best_action_indices[0]:
                    policy[action, state[0], state[1]] = (1 - epsilon) / len(best_action_indices[0]) + epsilon / 4
                else:
                    policy[action, state[0], state[1]] = epsilon / 4
    return policy

# The robot epoch function for single processing
def robot_epoch(robot):
    # load MC model from class
    # gamma=0.6,epsilon=0.2 are optimal for app.py
    model_free = MC(robot,gamma=0.6,epsilon=0.2,max_iteration=120) 
    # the optimal policy after Monte Carlo
    optimal_policy = on_policy_mc_control(model_free)
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    # indices of direction with the highest policy value
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    # epsilon-greedy
    for index in range(0, 4):
        if index in indices:
            probability.append(1/len(indices))
        else:
            probability.append(0)
    direction = np.random.choice(model_free.directions, p=probability)
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()

# The robot epoch function for multiple processing
def robot_epoch_(robot,gamma,epsilon):
    # load MC model from class
    model_free = MC(robot,gamma,epsilon,max_iteration=100)# '100': here to set the 200 iterations to get the heatmap in our report.
    # the optimal policy after Monte Carlo
    optimal_policy = on_policy_mc_control(model_free)
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    # indices of direction with the highest policy value
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    # epsilon-greedy
    for index in range(0, 4):
        if index in indices:
            probability.append(1/len(indices))
        else:
            probability.append(0)
    direction = np.random.choice(model_free.directions, p=probability)
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()

# The robot_epoch_a function for headless average
def robot_epoch_a(robot):
    # load MC model from class
    model_free = MC(robot,gamma=0.6,epsilon=0.2,max_iteration=50)# '50': here to set 200 to get the table results in our report.
    # the optimal policy after Monte Carlo
    optimal_policy = on_policy_mc_control(model_free)
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    # indices of direction with the highest policy value
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    # epsilon-greedy
    for index in range(0, 4):
        if index in indices:
            probability.append(1/len(indices))
        else:
            probability.append(0)
    direction = np.random.choice(model_free.directions, p=probability)
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()
