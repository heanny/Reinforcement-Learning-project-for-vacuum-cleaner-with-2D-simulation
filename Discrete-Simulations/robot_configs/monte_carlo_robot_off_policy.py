from copy import deepcopy
from mc import MC
import numpy as np

def create_greedy_policy(Q):
    """
    Creates a greedy policy based on Q values.
    Args:
        Q: A dictionary that maps from state -> action values
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """
    A = np.zeros_like(Q, dtype=float)
    best_action = np.argmax(Q)
    A[best_action] = 1.0
    return A


def off_policy_mc_control(self):
    """
    using the off policy Monte Carlo
    Args:
        MC class for Monte Carlo
    Returns:
        The target policy
    """
    # initialization the Q_table, the behavior policy and target policy
    Q_tmp = deepcopy(self.Q)
    policy_behavior = self.policy
    policy_target = create_greedy_policy(Q_tmp)
    # get numerator and denominator of Q(s,a)
    D = self.D
    N = self.N
    episode_new = []
    # repeat for l iterations:
    for l in range(int(self.max_iteration)):
        # generate an episode
        episode = self.generate_episode(policy_behavior)
        # set the initial omega value as 1.0
        omega = 1.0
        # for each pair of state and action appearing in the episode at time tau or later:
        for item in episode:
            state = item[0]
            action = self.direction_index_map[item[1]]
            reward = item[2]
            # get the action list for the current state
            action_list = policy_target[:, state[0], state[1]]
            # select a greedy action for current state
            greedy_action_for_target_pi = np.argmax(action_list)
            action_list_indices = np.where(action_list == policy_target[greedy_action_for_target_pi, state[0], state[1]])

            # Choose time tau with the latest time at action_tau != behavior_policy(state_tau)
            for t in reversed(range(len(episode))):
                if action in action_list_indices[0]:
                    t -= 1
                else:
                    tau = t
                    for tau in range(tau, len(episode)):
                        N = N + omega * reward
                        D = D + omega

                    # calculate the new Q(s,a) by numerator and denominator of Q(s,a)
                    episode_tmp = (state[0], state[1]), self.directions[action], N/D
                    episode_new.append(episode_tmp)

            # Generate Q_table based on sampling
            self.Q_table(episode_new)
            Q_tmp = self.Q

            # for each state, we find the action which argmax the Q(s,a) as the policy for this state
            A = Q_tmp[:, state[0], state[1]]
            best_action = np.argmax(A)
            best_action_indices = np.where(A == Q_tmp[best_action, state[0], state[1]])

            # update omega value based on the behavior policy and it will be multiplied to update N and D
            omega *= 1/policy_behavior[action, state[0], state[1]]
            policy_new = [0, 0, 0, 0]

            # if the best action is not the only one, then each optimal direction has the same probability
            prob = 1 / len(best_action_indices[0])
            # check the convergence
            if action not in best_action_indices[0]:
                break
            # if not, update the target policy
            else:
                for action in best_action_indices[0]:
                    policy_new[action] = prob
                policy_target[:, state[0], state[1]] = policy_new
    return policy_target


def robot_epoch(robot):
    """
    propose the move of the robot based on the returned policy
    Args:
        robot: the robot of our environment
    """
    model_free = MC(robot,gamma=0.45, max_iteration=200)
    # the optimal policy is the policy we get from off-policy MC algorithm
    optimal_policy = off_policy_mc_control(model_free)
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
    # we propose the move based on the optimal policy with equal probability for each direction if each state has more
    # than one action of the optimal policy
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

# we use robot_epoch_ to run the multiple processors headless.
def robot_epoch_(robot,gamma):
    """
    propose the move of the robot based on the returned policy
    Args:
        robot: the robot of our environment
    """
    model_free = MC(robot, gamma, max_iteration=200)
    optimal_policy = off_policy_mc_control(model_free)
    policy_of_current_state = optimal_policy[:, robot.pos[0], robot.pos[1]]
    indices = np.where(policy_of_current_state == np.max(policy_of_current_state))[0]
    probability = []
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