import copy
import numpy as np
from numpy.random import choice
from td import TD

def sarsa(model_free, alpha, gamma, epsilon, episodes):
    """
    This function implement the TD on-policy control(sarsa)
    :param model_free: temporal difference class TD from td.py
    :param alpha: the learning rate
    :param gamma: discounted factor
    :param epsilon: epsilon-greedy factor
    :returns policy: the optimal policy matrix
    """
    while episodes:
        count = np.zeros((model_free.n_rows, model_free.n_cols))
        robot_copy = copy.deepcopy(model_free.robot)
        done = False
        # initial state, policy and action
        # the policy format: policy[action, state[0], state[1]]
        state = robot_copy.pos
        policy = model_free.policy[:, state[0], state[1]]
        action = choice(model_free.directions, p=policy)

        # When robot is alive and the 100% cleanliness is not done
        # In case of the robot will check the visited tiles multiple times in one episode,
        # we set a counter named count to control the repeated visiting behaviors.
        while robot_copy.alive and not done and np.max(count) < 3:
            # sample and get the next state, and the reward of the next state,
            # the action of the next state from the policy.
            state_, reward = model_free.simulation(robot_copy, action)
            policy_ = model_free.policy[:, state_[0], state_[1]]
            action_ = choice(model_free.directions, p=policy_)
            count[state[0]][state[1]] += 1

            # update Q table based on the current action, the current state, the next action, the next state
            model_free.update_Qvalue(action, state, state_, reward, alpha, gamma, True, action_)
            # update policy based on the current state
            model_free.update_policy(epsilon, state)

            # use the next action, state as the current action and state to repeat sampling.
            action = action_
            state = state_

            # if the cleanliness percentage is 100, our goal is reached.
            clean = (robot_copy.grid.cells == 0).sum()
            dirty = (robot_copy.grid.cells >= 1).sum()
            cleanliness = clean / (clean + dirty)
            if cleanliness == 1:
                done = True
        episodes -= 1
    return model_free.policy


def robot_epoch(robot):
    # initial TD class
    model_free = TD(robot)
    # The optimal parameter is alpha=0.1, gamma=1.0, epsilon=0.2 for app.py
    # the optimal policy is the policy we get from SARSA algorithm
    optimal_policy = sarsa(model_free, 0.1, 1.0, 0.2, 500)
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    # we propose the move based on the optimal policy with equal probability for each direction if each state has more
    # than one action of the optimal policy
    for index in range(0, 4):
        if index in indices:
            probability.append(1 / len(indices))
        else:
            probability.append(0)
    direction = choice(model_free.directions, p=probability)
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()


# robot_epoch_ function is used for running the multiple processor headless file
def robot_epoch_(robot, lr, gamma, epsilon):
    model_free = TD(robot)
    # the optimal policy is the policy we get from SARSA algorithm
    optimal_policy = sarsa(model_free, lr, gamma, epsilon, 200)# '200': here to set the 500 iterations to get the heatmap of the report.
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    # we propose the move based on the optimal policy with equal probability for each direction if each state has more
    # than one action of the optimal policy
    for index in range(0, 4):
        if index in indices:
            probability.append(1 / len(indices))
        else:
            probability.append(0)
    direction = choice(model_free.directions, p=probability)
    while not direction == robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()
