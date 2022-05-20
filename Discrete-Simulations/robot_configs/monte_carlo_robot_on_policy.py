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
        policy_stable=True
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
                old_policy = policy[action, state[0], state[1]]
                if action in best_action_indices[0]:
                    policy[action, state[0], state[1]] = (1 - epsilon) / len(best_action_indices[0]) + epsilon / 4
                else:
                    policy[action, state[0], state[1]] = epsilon / 4
                if policy[action, state[0], state[1]] != old_policy:
                    policy_stable=False
        if  policy_stable== True:
            print("converged")
            print(l)
            break    
    return policy

# The robot epoch function for single processing
def robot_epoch(robot):
    # load MC model from class
    model_free = MC(robot,gamma=0.6,epsilon=0.6,max_iteration=200)
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
    model_free = MC(robot,gamma,epsilon,max_iteration=200)
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