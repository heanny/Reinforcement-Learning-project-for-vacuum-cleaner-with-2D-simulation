from tkinter.tix import DirSelectBox
import numpy as np
import copy
from collections import defaultdict


# initialization function
def init_Q(n_rows, n_cols,dirs):
    return np.full((n_rows, n_cols,4),float('-inf'))

def make_epsilon_greedy_policy(n_rows,n_cols, Q, rewards,policy, num_actions=4,epsilon=0.1):
    def policy_for_state(i,j):
        A = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[i,j,:])
        A[best_action] += (1.0 - epsilon)
        return A
    
    for i in range(0,n_rows):
        for j in range(0,n_cols):
            if rewards[i][j] not in [-3,-1]:
                policy[i][j]=policy_for_state(i,j)
    return policy
    

def get_current_rewards(cells, transformation):
    """
    Get the reward matrix based on grid's current circumstances(each tile's label) and robot's history.
    :param cells: cells attribute of robot.grid, a matrix record the label of each tile
    :param transformation: a punishment matrix, where each element is the punishment of each tile
    :returns combined_reward: a reward matrix
    """
    reward = copy.deepcopy(cells)
    # label < -2L: this tile has a robot with different direction inside it. We set it to 0, meaning it is already clean.
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

def episode_generation(i,j,dirs,rewards,policy):
    # Generate an episode.
    # An episode is an array of (state, action, reward) tuples
    episode = []
    dir_keys = dirs.get_keys()
    state=(i,j)
    done=False
    for t in range(1e9):
        probs = policy(state[0],state[1])
        action = np.random.choice(np.arange(len(probs)), p=probs)
        action_name =dir_keys[action]
        # get the corresponding coordinates for an action e.g. 0 -> "n" -> row -1, column remains the same
        direction=reversed(dirs[action_name]) # reverse to get the correct value for i and j
        next_state = state+direction
        # TODO: What if it hits obstacle or death tile
        # if policy(next_state[0]next_state,[1]) is not :
        #     done=True
        reward=rewards(i,j)
        state=(i,j)
        episode.append((state, action_name, reward))
        if done: # if the evaluation end
            break # TODO
        state = next_state
    return episode

def Q_table(episode,Q,gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    sa_in_episode = set([(x[0], x[1]) for x in episode])
    for state, action in sa_in_episode:
        sa_pair = (state, action)
        # Calculate Q(s,a) for each (s,a) pair (mc policy evaluation)
        # Find the first occurance of the (state, action) pair in the episode
        first_occurence_idx = next(i for i,x in enumerate(episode)
                                    if x[0] == state and x[1] == action)
        # Sum up all rewards since the first occurance
        G = sum([x[2]*(gamma**i) for i,x in enumerate(episode[first_occurence_idx:])])
        # Calculate average return for this state over all sampled episodes
        returns_sum[sa_pair] += G
        returns_count[sa_pair] += 1.0
        Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
    return Q


def on_policy_mc_control(robot,dirs,rewards,max_iterations=10):
    n_cols = robot.grid.n_rows 
    n_rows = robot.grid.n_cols
    # initialization
    Q=init_Q(n_rows, n_cols,dirs)
    # first set the policy to be zero
    policy = np.full((n_rows, n_cols), 0)       

    # repeat till value converge:
    for l in range (int(max_iterations)):
        # generate an episode
        policy=make_epsilon_greedy_policy(n_rows,n_cols,Q,rewards,policy,num_actions=4,epsilon=0.1)
        episode = episode_generation(i=robot.pos[0],j=robot.pos[1],policy=policy)
        Q = Q_table(episode,Q)
    return 0

def robot_epoch(robot):
    # initialize global variable "history" when starting the robot, to add the cleaned tiles in a matrix
    if not any(robot.history):
        n_cols = robot.grid.n_rows
        n_rows = robot.grid.n_cols
        global history
        history = np.full((n_rows, n_cols),0.0)
    history=np.where(history<99,history,99)
    transformation = np.where(history==0, history, -0.01*history) # the range of each element is (-1,0]
    rewards = get_current_rewards(robot.grid.cells,transformation)
    optimal_policy = on_policy_mc_control(robot,dirs,rewards)
