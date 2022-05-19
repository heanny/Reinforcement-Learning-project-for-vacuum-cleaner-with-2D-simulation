import multiprocessing
import json
# from robot_configs.sarsa_robot import robot_epoch_
from robot_configs.Q_learning_robot import robot_epoch_
# from robot_configs.monte_carlo_robot_on_policy import robot_epoch_
import pickle
import time
import numpy as np
from environment import Robot
import matplotlib.pyplot as plt
import seaborn as sns

# !!!please set False if you use monte_carlo_on_policy_robot!!!
TD_algo = True

# Here we use learning rate = 0.1 to test the gamma and epsilon for sarsa and Q_learning
# The parameter tuning for learning rate is in headless_alpha.py

def experiment(lr, gamma, epsilon, procnum, return_dict):
    #print("parameter setting:", item)
    grid_file = 'snake.grid'
    # Cleaned tile percentage at which the room is considered 'clean':
    stopping_criteria = 100
    # Keep track of some statistics:
    efficiencies = []
    n_moves = []
    deaths = 0
    cleaned = []
    # Run 100 times:
    start_time = time.time()
    for i in range(50):
    # Open the grid file.
    # (You can create one yourself using the provided editor).
        with open(f'grid_configs/{grid_file}', 'rb') as f:
            grid = pickle.load(f)
        # Calculate the total visitable tiles:
        n_total_tiles = (grid.cells >= 0).sum()
        # Spawn the robot at (1,1) facing north with battery drainage enabled:
        robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.5, battery_drain_lam=2)
        # Keep track of the number of robot decision epochs:
        n_epochs = 0
        while True:
            n_epochs += 1
            # Do a robot epoch (basically call the robot algorithm once):
            # check if we use the TD algorithm or Monte Carlo
            if TD_algo == True:
                robot_epoch_(robot, lr, gamma, epsilon)
            else:
                robot_epoch_(robot, gamma, epsilon)
            # Stop this simulation instance if robot died :( :
            if not robot.alive:
                deaths += 1
                break
            # Calculate some statistics:
            clean = (grid.cells == 0).sum()
            dirty = (grid.cells >= 1).sum()
            goal = (grid.cells == 2).sum()
            # Calculate the cleaned percentage:
            clean_percent = (clean / (dirty + clean)) * 100
            # See if the room can be considered clean, if so, stop the simulaiton instance:
            if clean_percent >= stopping_criteria and goal == 0:
                break
            # Calculate the effiency score:
            moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
            u_moves = set(moves)
            n_revisted_tiles = len(moves) - len(u_moves)
            efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)
        # Keep track of the last statistics for each simulation instance:
        efficiencies.append(float(efficiency))
        n_moves.append(len(robot.history[0]))
        cleaned.append(clean_percent)
    end_time = time.time()
    # calculate the average time (per epoch), average clean percentage and average efficiency
    average_time = (end_time-start_time)/(50*60)
    average_clean = np.mean(cleaned)
    average_eff = np.mean(efficiencies)
    return_dict[procnum] = np.round([average_time, average_clean, average_eff], decimals=2)

if __name__ == '__main__':
    # set the multiple processors
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    # save the efficiency values for all parameters as a list
    time_list = []
    clean_list = []
    eff_list = []
    # import the parameter json file
    f = open('parameters_tuning_pair.json')
    para_data = json.load(f)
    para = para_data['parameters']
    epsilon_info = ['0.0', '0.2', '0.4', '0.6', '0.8']
    gamma_info = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']

    for procnum in range(len(para)):
            p = multiprocessing.Process(target=experiment, args=(0.1, para[procnum]['gamma'], para[procnum]['epsilon'], procnum, return_dict))
            processes.append(p)
            p.start()

    for proc in processes:
        proc.join()

    sorted_return_dict = dict(sorted(return_dict.items()))
    a = sorted_return_dict
    for i in range(len(a)):
        time_list.append(a[i][2])
        clean_list.append(a[i][2])
        eff_list.append(a[i][2])

    # save results as the matrix format
    time_matrix = np.array(time_list).reshape((6, 5))
    clean_matrix = np.array(clean_list).reshape((6, 5))
    eff_matrix = np.array(eff_list).reshape((6, 5))

    # plot the heatmap for efficiency
    ax = sns.heatmap(eff_matrix, xticklabels=epsilon_info, yticklabels=gamma_info)
    ax.set(xlabel='Epsilon', ylabel='Gamma')
    ax.set_title('Efficiency(%)')
    plt.show()
