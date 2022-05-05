# Import our robot algorithm to use in this simulation:
import robot_configs.policy_iteration_robot as policy
import robot_configs.value_iteration_robot as value
import pickle
from environment import Robot
import matplotlib.pyplot as plt

grid_file = 'death.grid'
# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100

# Keep track of some statistics:
efficiencies_policy = []
n_moves_policy = []
deaths_policy = 0
cleaned_policy = []

efficiencies_value = []
n_moves_value = []
deaths_value = 0
cleaned_value = []

# Run 100 times:
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
        policy.robot_epoch(robot)
        # Stop this simulation instance if robot died :( :
        if not robot.alive:
            deaths_policy += 1
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
    efficiencies_policy.append(float(efficiency))
    n_moves_policy.append(len(robot.history[0]))
    cleaned_policy.append(clean_percent)

# Run 100 times:
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
        value.robot_epoch(robot)
        # Stop this simulation instance if robot died :( :
        if not robot.alive:
            deaths_value += 1
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
    efficiencies_value.append(float(efficiency))
    n_moves_value.append(len(robot.history[0]))
    cleaned_value.append(clean_percent)

# plots:
plt.figure(figsize=(8,6))
plt.hist(cleaned_policy, alpha=0.5, label="policy iteration")
plt.hist(cleaned_value, alpha=0.5, label="value iteration")
plt.xlabel("Cleaned Percentage", size=14)
plt.ylabel("Count", size=14)
plt.title("Comparison of Cleaned Percentage Between Two Algorithms")
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(8,6))
plt.hist(efficiencies_policy, alpha=0.5, label="policy iteration")
plt.hist(efficiencies_value, alpha=0.5, label="value iteration")
plt.xlabel("efficiencies Percentage", size=14)
plt.ylabel("Count", size=14)
plt.title("Comparison of efficiencies Percentage Between Two Algorithms")
plt.legend(loc='upper right')
plt.show()
