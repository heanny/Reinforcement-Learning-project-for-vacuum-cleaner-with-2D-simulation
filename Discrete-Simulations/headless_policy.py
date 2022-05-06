# Import our robot algorithm to use in this simulation:
import numpy as np

from robot_configs.policy_iteration_robot_1 import robot_epoch as robot_epoch1
from robot_configs.policy_iteration_robot_2 import robot_epoch as robot_epoch2
from robot_configs.policy_iteration_robot_3 import robot_epoch as robot_epoch3
import pickle
from environment import Robot
import matplotlib.pyplot as plt

grid_file = 'house.grid'
# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100

# Keep track of some statistics for robot1:
efficiencies1 = []
n_moves1 = []
deaths1 = 0
cleaned1 = []

# Keep track of some statistics for robot1:
efficiencies2 = []
n_moves2 = []
deaths2 = 0
cleaned2 = []

# Keep track of some statistics for robot1:
efficiencies3 = []
n_moves3 = []
deaths3 = 0
cleaned3 = []

# For theta=0.51, run 100 times:
for i in range(100):
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
        robot_epoch1(robot)
        # Stop this simulation instance if robot died :( :
        if not robot.alive:
            deaths1 += 1
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
    efficiencies1.append(float(efficiency))
    n_moves1.append(len(robot.history[0]))
    cleaned1.append(clean_percent)

# For theta=2.1, run 100 times:
for i in range(100):
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
        robot_epoch2(robot)
        # Stop this simulation instance if robot died :( :
        if not robot.alive:
            deaths2 += 1
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
    efficiencies2.append(float(efficiency))
    n_moves2.append(len(robot.history[0]))
    cleaned2.append(clean_percent)

# For theta=5, run 100 times:
for i in range(100):
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
        robot_epoch3(robot)
        # Stop this simulation instance if robot died :( :
        if not robot.alive:
            deaths3 += 1
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
    efficiencies3.append(float(efficiency))
    n_moves3.append(len(robot.history[0]))
    cleaned3.append(clean_percent)


# # Make some plots:
# plt.hist(cleaned)
# plt.title('Percentage of tiles cleaned.')
# plt.xlabel('% cleaned')
# plt.ylabel('count')
# plt.show()
#
# plt.hist(efficiencies)
# plt.title('Efficiency of robot.')
# plt.xlabel('Efficiency %')
# plt.ylabel('count')
# plt.show()

# Cleaning plots:
scale = lambda x,y:(x+y)/2
res_cl1 = np.histogram(cleaned1,bins=10)
res_cl1_x = scale(res_cl1[1][:10],res_cl1[1][1:11])
# print(f"cl1:{res_cl1[0]} \n {res_cl1[1]} \n {test_x}")

res_cl2 = np.histogram(cleaned2,bins=10)
res_cl2_x = scale(res_cl2[1][:10],res_cl2[1][1:11])
res_cl3 = np.histogram(cleaned3,bins=10)
res_cl3_x = scale(res_cl3[1][:10],res_cl3[1][1:11])

plt.plot(res_cl1_x,res_cl1[0], marker='o', markersize=3)
plt.plot(res_cl2_x,res_cl2[0], marker='o', markersize=3)
plt.plot(res_cl3_x,res_cl3[0], marker='o', markersize=3)
plt.title('Percentage of tiles cleaned with gamma=1 and battery constrain')
plt.xlabel('% cleaned')
plt.ylabel('count')

for x,y in zip(res_cl1_x,res_cl1[0]):
    plt.text(x,y,y, ha='center', va='bottom', fontsize=10)
for x,y in zip(res_cl2_x,res_cl2[0]):
    plt.text(x,y,y, ha='center', va='bottom', fontsize=10)
for x,y in zip(res_cl3_x,res_cl3[0]):
    plt.text(x,y,y, ha='center', va='bottom', fontsize=10)
plt.legend(['threshold=2.1', 'threshold=50', 'threshold=200'])
plt.show()

# Efficiency plots:
scale = lambda x,y:(x+y)/2
res_ef1 = np.histogram(efficiencies1,bins=10)
res_ef1_x = scale(res_ef1[1][:10],res_ef1[1][1:11])
# print(f"cl1:{res_cl1[0]} \n {res_cl1[1]} \n {test_x}")

res_ef2 = np.histogram(efficiencies2,bins=10)
res_ef2_x = scale(res_ef2[1][:10],res_ef2[1][1:11])

res_ef3 = np.histogram(efficiencies3,bins=10)
res_ef3_x = scale(res_ef3[1][:10],res_ef3[1][1:11])

plt.plot(res_ef1_x,res_ef1[0], marker='o', markersize=3)
plt.plot(res_ef2_x,res_ef2[0], marker='o', markersize=3)
plt.plot(res_ef3_x,res_ef3[0], marker='o', markersize=3)
plt.title('Efficiency of robot with gamma=1 and battery constrain')
plt.xlabel('Efficiency %')
plt.ylabel('count')

for x,y in zip(res_ef1_x,res_ef1[0]):
    plt.text(x,y,y, ha='center', va='bottom', fontsize=10)
for x,y in zip(res_ef2_x,res_ef2[0]):
    plt.text(x,y,y, ha='center', va='bottom', fontsize=10)
for x,y in zip(res_ef3_x,res_ef3[0]):
    plt.text(x,y,y, ha='center', va='bottom', fontsize=10)
plt.legend(['threshold=2.1', 'threshold=50', 'threshold=200'])
plt.show()