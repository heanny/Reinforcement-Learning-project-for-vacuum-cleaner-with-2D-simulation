# Import our robot algorithm to use in this simulation:
from robot_configs.monte_carlo_robot_off_policy import robot_epoch
#from robot_configs.sarsa_robot import robot_epoch
#from robot_configs.Q_learning_robot import robot_epoch
#from robot_configs.policy_iteration_robot import robot_epoch
#from robot_configs.value_iteration_robot import robot_epoch
#from robot_configs.monte_carlo_robot_on_policy import robot_epoch
import pickle
import time
import numpy as np
from environment import Robot

# This headless is used for getting the average efficiency and runtime of each robot on house grid
# Please uncomment the robot you would like to test and comment other robots out.
# If you would like to try monte carlo robot, please go to line 30 and 38 to change the battery setting with few efforts.
# Note that the off-policy MC robot may take more than 2 hours to finish, but our report results are valid since we test 50 runs per robot.

grid_file = 'house.grid'
# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100

# Keep track of some statistics:
efficiencies = []
n_moves = []
deaths = 0
cleaned = []

# Run 50 times:
start_time = time.time()
for i in range(50):# change "range(20)" in line 30 for MC robots for fewer runs to show the results.
    # Open the grid file.
    # (You can create one yourself using the provided editor).
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # Calculate the total visitable tiles:
    n_total_tiles = (grid.cells >= 0).sum()
    # Spawn the robot at (1,1) facing north with battery drainage enabled:
    robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0, battery_drain_lam=0) #please set battery_drain_p=0.5, battery_drain_lam=2.0 to make sure that one run stops in a short time. 
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
average_time = (end_time-start_time)/(50*60)
average_clean = np.mean(cleaned)
average_eff = np.mean(efficiencies)
print("average_time:",average_time)
print("average_clean:",average_clean)
print("average_eff:",average_eff)


