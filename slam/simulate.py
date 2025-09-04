import matplotlib.pyplot as plt
from slam.gridmap import OccupancyGrid
from slam.sensors import inverse_sensor_model, sense_and_update
from slam.robot import Robot
from slam.maze import create_maze
import numpy as np

def run_demo():
    maze = create_maze()
    grid = OccupancyGrid(maze.shape[1], maze.shape[0])
    robot = Robot(1, 1, 0.0)
    goal  = (8, 8)

    plt.ion()

    for step in range(15):
        # sense environment with limited lidar
        log_odds = sense_and_update(maze, robot.pose(), grid.log_odds)

        # move policy: east if free, else south
        if maze[robot.y, robot.x+1] == 0:
            robot.move(1, 0)
        else:
            robot.move(0, 1)

        # visualize
        plt.subplot(1,2,1)
        plt.imshow(maze, cmap="gray_r", origin="lower")
        plt.plot(robot.x, robot.y, "ro"); plt.plot(goal[0], goal[1], "gx")
        plt.title("Ground Truth Maze")

        plt.subplot(1,2,2)
        plt.imshow(grid.prob_map(), cmap="gray", origin="lower", vmin=0, vmax=1)
        plt.plot(robot.x, robot.y, "ro"); plt.plot(goal[0], goal[1], "gx")
        plt.title("Estimated Map")

        plt.pause(0.5); plt.clf()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_demo()