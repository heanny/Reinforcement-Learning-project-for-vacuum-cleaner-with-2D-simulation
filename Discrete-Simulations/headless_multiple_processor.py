import multiprocessing
import json
#from robot_configs.sarsa_robot import robot_epoch_
#from robot_configs.Q_learning_robot import robot_epoch_
#from robot_configs.monte_carlo_robot_on_policy import robot_epoch_
from robot_configs.monte_carlo_robot_off_policy import robot_epoch_
import pickle
import time
import numpy as np
from environment import Robot
import matplotlib.pyplot as plt
import seaborn as sns

# Please read the README before using this to generate the heatmap and line chart.
# We did our experiments on Macbook Pro with M1 pro chip, which has 10-core CPU and 16-core GPU.
# Tuning the gamma and epsilon for Q-learning and Sarsa with single_para_flag == False and TD_algo == True
# Tuning the learning rate for Q-learning and Sarsa with single_para_flag == Ture and TD_algo == True
# Tuning the gamma and epsilon for Monte Carlo on-policy version with single_para_flag == True and TD_algo == False
# Tuning the gamma for Monte Carlo off-policy version with single_para_flag == False and TD_algo == False

TD_algo = False
single_para_flag = True

def experiment(lr, gamma, epsilon, procnum, return_dict):
    """
    do the experiment with given parameters

    :param lr: learning rate for TD algorithms
    :param gamma: discounted factor
    :param epsilon: epsilon-greedy factor
    :param procnum: shared variable to communicate peocessors
    :param return_dict: save the results as dictionary
    """
    grid_file = 'snake.grid'
    # Cleaned tile percentage at which the room is considered 'clean':
    stopping_criteria = 100
    # Keep track of some statistics:
    efficiencies = []
    n_moves = []
    deaths = 0
    cleaned = []

    start_time = time.time()
    # 50 runs per robot
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

            # Tuning the gamma and epsilon for Q-learning and Sarsa
            if single_para_flag == False and TD_algo == True:
                # Do a robot epoch (basically call the robot algorithm once):
                robot_epoch_(robot, lr, gamma, epsilon)
                if not robot.alive:
                    deaths += 1
                    break

            # Tuning the learning rate for Q-learning and Sarsa
            elif single_para_flag == True and TD_algo == True:
                # Do a robot epoch (basically call the robot algorithm once):
                robot_epoch_(robot, lr, gamma, epsilon)
                # Stop this simulation instance if robot died :(
                if not robot.alive:
                    deaths += 1
                    break

            # Tuning the gamma and epsilon for Monte Carlo on policy version
            elif single_para_flag == False and TD_algo == False:
                robot_epoch_(robot, gamma, epsilon)
                if not robot.alive:
                    deaths += 1
                    break

            # Tuning the gamma for Monte Carlo off policy version
            else:
                robot_epoch_(robot, gamma)
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

    # save the efficiency, runtime, cleanliness results for all parameters
    time_list = []
    clean_list = []
    eff_list = []

    # import the parameter json file
    f = open('parameters_tuning_pair.json')
    para_data = json.load(f)
    para = para_data['pair_parameters']
    para_lr_sarsa = para_data['learning_rate_sarsa']
    para_lr_Q = para_data['learning_rate_Q']
    para_gamma_op = para_data['gamma_off_policy_MC']

    # the index list for heatmap and linechart
    epsilon_info = ['0.0', '0.2', '0.4', '0.6', '0.8']
    gamma_info = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
    alpha_info = ['0.1', '0.2', '0.3', '0.4', '0.5']

    # Tuning the gamma and epsilon for Q-learning and Sarsa
    if single_para_flag == False and TD_algo == True:
        for procnum in range(len(para)):
            p = multiprocessing.Process(target=experiment, args=(0.1, para[procnum]['gamma'], para[procnum]['epsilon'], procnum, return_dict))
            processes.append(p)
            p.start()

    # Tuning the learning rate for Q-learning and Sarsa
    elif single_para_flag == True and TD_algo == True:
        for procnum in range(len(para_lr_Q)):
            p = multiprocessing.Process(target=experiment, args=(para_lr_Q[procnum]['lr'], para_lr_Q[procnum]['gamma'], para_lr_Q[procnum]['epsilon'], procnum, return_dict))
            processes.append(p)
            p.start()

    # Tuning the gamma and epsilon for Monte Carlo on-policy version
    elif single_para_flag == False and TD_algo == False:
        for procnum in range(len(para)):
            p = multiprocessing.Process(target=experiment, args=(0, para[procnum]['gamma'], para[procnum]['epsilon'], procnum, return_dict))
            processes.append(p)
            p.start()

    # Tuning the gamma for Monte Carlo off-policy version
    else:
        for procnum in range(len(para_gamma_op)):
            p = multiprocessing.Process(target=experiment, args=(0, para_gamma_op[procnum]['gamma'], 0, procnum, return_dict))
            processes.append(p)
            p.start()

    for proc in processes:
        proc.join()

    if single_para_flag == False:
        # plot the heatmap for efficiency of gamma and epsilon tuning for on-policy MC
        # Sort the results since processors finish at different time
        sorted_return_dict = dict(sorted(return_dict.items()))
        a = sorted_return_dict
        for i in range(len(a)):
            time_list.append(a[i][0])
            clean_list.append(a[i][1])
            eff_list.append(a[i][2])

        # save results as the matrix format
        time_matrix = np.array(time_list).reshape((6, 5))
        print("average_time:", time_matrix)
        clean_matrix = np.array(clean_list).reshape((6, 5))
        print("average_clean:", clean_matrix)
        eff_matrix = np.array(eff_list).reshape((6, 5))

        # plot the heatmap for efficiency
        ax = sns.heatmap(eff_matrix, xticklabels=epsilon_info, yticklabels=gamma_info, annot=True, cmap="Blues")
        ax.set(xlabel='Epsilon', ylabel='Gamma')
        ax.set_title('Efficiency(%)')
        plt.show()
    else:
        if TD_algo == False:
            # plot the linechart for gamma tuning of off-policy MC
            sorted_return_dict = dict(sorted(return_dict.items()))
            a = sorted_return_dict
            for i in range(len(a)):
                time_list.append(a[i][0])
                clean_list.append(a[i][1])
                eff_list.append(a[i][2])
            print("average_time", time_list)
            print("average_cleanliness", clean_list)
            sns.set_theme()
            sns.set_context("paper")
            ax = sns.lineplot(x=gamma_info, y=eff_list, marker='o')
            ax.set(xlabel='Gamma', ylabel='efficiency(%)')
            # Annotate the label for data points
            for i, point in enumerate(eff_list):
                plt.annotate(point, (gamma_info[i], eff_list[i]))
            ax.set_title('Efficiency(%)')
            plt.show()

        else:
            # plot the linechart for learning rate tuning
            sorted_return_dict = dict(sorted(return_dict.items()))
            a = sorted_return_dict
            for i in range(len(a)):
                time_list.append(a[i][0])
                clean_list.append(a[i][1])
                eff_list.append(a[i][2])
            print("average_time", time_list)
            print("average_cleanliness", clean_list)
            sns.set_theme()
            sns.set_context("paper")
            ax = sns.lineplot(x=alpha_info, y=eff_list, marker='o')
            ax.set(xlabel='learning rate(alpha)', ylabel='efficiency(%)')
            # Annotate the label for data points
            for i, point in enumerate(eff_list):
                plt.annotate(point, (alpha_info[i], eff_list[i]))
            ax.set_title('Efficiency(%) for different learning rates')
            plt.show()
