from robot_configs.monte_carlo_robot_off_policy import robot_epoch_a # delete"_a" for original experiments settings which takes longer time
#from robot_configs.monte_carlo_robot_on_policy import robot_epoch_a # delete"_a" for original experiments settings which takes longer time
#from robot_configs.sarsa_robot import robot_epoch_a # delete"_a" for original experiments settings which takes longer time
#from robot_configs.Q_learning_robot import robot_epoch_a # delete"_a" for original experiments settings which takes longer time
#from robot_configs.policy_iteration_robot import robot_epoch
#from robot_configs.value_iteration_robot import robot_epoch
import pickle
import time
import numpy as np
from environment import Robot

# Please delete"_a" for each "robot_epoch_a" if you would like to run our original settings for experiments, which will take longer to finish.
# This headless is used for getting the average efficiency and runtime of each robot on house grid
# Please uncomment the robot you would like to test and comment other robots out.
# If you would like to try monte carlo robot, please go to line 28 and 38 to change the battery setting and runs with few efforts.
# Note that the off-policy MC robot may take more than 2 hours to finish, but our report results are valid since we test 50 runs per robot.

grid_file = 'house.grid'
# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100
# Keep track of some statistics:
efficiencies = []
n_moves = []
deaths = 0
cleaned = []
# 50 runs per robot
start_time = time.time()
n = 50 # change "n = 10" in line 28 for MC robots for fewer runs to show the results.
for i in range(n):# change "range(10)" in line 30 for MC robots for fewer runs to show the results.
    # Open the grid file.
    # (You can create one yourself using the provided editor).
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # Calculate the total visitable tiles:
    n_total_tiles = (grid.cells >= 0).sum()
    # Spawn the robot at (1,1) facing north with battery drainage enabled:
    #please set battery_drain_p=0.5, battery_drain_lam=1.0 in line 38 to make sure that one run stops in a short time. 
    robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0, battery_drain_lam=0) 
    # Keep track of the number of robot decision epochs:
    n_epochs = 0
    while True:
        n_epochs += 1
        # Do a robot epoch (basically call the robot algorithm once):
        robot_epoch(robot)
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
    print(i+1,'run are done.')
end_time = time.time()

#print out the average efficiency and runtime
average_time = (end_time-start_time)/(n*60)
average_clean = np.mean(cleaned)
average_eff = np.mean(efficiencies)
print("average_time:",average_time)
print("average_clean:",average_clean)
print("average_eff:",average_eff)


