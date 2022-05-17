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
    # initialization
    Q_tmp = deepcopy(self.Q)
    policy_behavior = self.policy
    policy_target = create_greedy_policy(Q_tmp)

    D = self.D
    N = self.N
    episode_new = []
    for l in range(int(self.max_iteration)):
        # generate an episode
        episode = self.generate_episode(policy_behavior)
        # set omega
        omega = 1.0
        # for each pair s,a appearing in the episode t time tau or later:
        for item in episode:
            state = item[0]
            action = self.direction_index_map[item[1]]
            reward = item[2]
            action_list = policy_target[:, state[0], state[1]]
            greedy_action_for_target_pi = np.argmax(action_list)
            action_list_indices = np.where(action_list == policy_target[greedy_action_for_target_pi, state[0], state[1]])
            # Choose tau with the latest time at action_tau != pi(state_tau)
            for t in reversed(range(len(episode))):
                if action in action_list_indices[0]:
                    t -= 1
                else:
                    tau = t
                    for tau in range(tau, len(episode)):
                        N = N + omega * reward
                        D = D + omega
                    # calculate the new Q(s,a) by N and D
                    episode_tmp = (state[0], state[1]), self.directions[action], N/D
                    episode_new.append(episode_tmp)
            Q_tmp = self.Q_table(episode_new)
            # for each state, we find the action which argmax the Q(s,a) as the policy for this state
            A = Q_tmp[:, state[0], state[1]]
            best_action = np.argmax(A)
            best_action_indices = np.where(A == Q_tmp[best_action, state[0], state[1]])
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
    model_free = MC(robot,gamma=0.45, max_iteration=500)
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