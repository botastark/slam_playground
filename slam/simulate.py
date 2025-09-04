import matplotlib.pyplot as plt
from slam.gridmap import OccupancyGrid
from slam.sensors import inverse_sensor_model, sense_and_update, cast_ray
from slam.robot import Robot, heading_label
from slam.maze import create_maze
from slam.astar import astar
import numpy as np
import math


def run_demo():
    maze = create_maze()
    start = (1, 8)
    goal = (8, 1)
    robot = Robot(*start, 0.0)
   # Run A*
    occ_grid = OccupancyGrid(maze.shape[1], maze.shape[0])
    path = astar(maze, start, goal, occ_threshold=0.5)  # you may need to adapt astar to use maze directly
    if not path:
        print("No path found!")
        return
    plt.figure()
    plt.ion()


    for next_pose in path[1:]:
        robot.follow_path(next_pose, maze)

        # Draw
        plt.imshow(maze, cmap="gray_r", origin="upper")
        plt.plot(robot.x, robot.y, "ro")

        # arrow
        plt.arrow(robot.x, robot.y,
                  0.5*math.cos(robot.theta),
                  0.5*math.sin(robot.theta),
                  head_width=0.2, head_length=0.2, fc='r', ec='r')

        # label
        plt.text(robot.x + 0.3, robot.y + 0.3, heading_label(robot.theta),
                 color="red", fontsize=12, weight="bold")

        # path
        px, py = zip(*path)
        plt.plot(px, py, "b.-")

        plt.pause(0.5)
        plt.clf()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_demo()