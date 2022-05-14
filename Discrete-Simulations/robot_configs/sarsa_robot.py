from sarsa_brain import SarsaTable
import copy
import numpy as np


def get_current_rewards(cells):
    """
    Get the reward matrix based on grid's current circumstances(each tile's label) and robot's history.
    :param cells: cells attribute of robot.grid, a matrix record the label of each tile
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
    print("reward")
    return reward


def robot_epoch(robot):
    # env = Maze()
    RL = SarsaTable(actions=list(range(4)))
    print("initial ok")
    for episode in range(100):
        # initial observation
        # observation = env.reset()
        observation = robot.pos

        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            # env.render()

            # RL take action and get next observation and reward
            while action != robot.orientation:
                # If we don't have the wanted orientation, rotate clockwise until we do:
                robot.rotate('r')
            # Move:
            robot.move()

            reward = get_current_rewards(robot.grid.cells)
            done = robot.alive
            if done == False:
                observation_ = 'terminal'
            else:
                observation_ = robot.pos
            # observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    # print('game over')
    # env.destroy()

# if __name__ == "__main__":
#     env = Maze()
#     RL = SarsaTable(actions=list(range(robot.orients)))


# env.after(100, update)
# env.mainloop()
