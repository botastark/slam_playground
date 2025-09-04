import matplotlib.pyplot as plt
from slam.gridmap import OccupancyGrid
from slam.sensors import inverse_sensor_model, sense_and_update, cast_ray
from slam.robot import Robot, heading_label
from slam.maze import create_maze
from slam.astar import astar
from slam.plot_state import plot_state
import numpy as np
import math

LIDAR_CONFIG = {"fov_deg": 90, "n_beams": 30, "max_range": 5}


def run_demo_fixed_path():
    maze = create_maze()
    start = (1, 8)
    goal = (8, 1)
    robot = Robot(*start, 0.0)

    # same fixed path as before (through the maze)
    # fixed_path = [
    #     (1, 8),
    #     (2, 8),
    #     (3, 8),
    #     (4, 8),
    #     (4, 7),
    #     (5, 7),
    #     (6, 7),
    #     (6, 6),
    #     (7, 6),
    #     (8, 6),
    #     (8, 5),
    #     (8, 4),
    #     (8, 3),
    #     (8, 2),
    #     (8, 1),
    # ]
    fixed_path = [
        (1, 8),
        (2, 8),
        (3, 8),
        (4, 8),
        (4, 7),
        (5, 7),
        (6, 7),
        (6, 6),
        (7, 6),
        (8, 6),
        (8, 5),
        (8, 4),
        (8, 3),
        (8, 2),
        (8, 1),
    ]

    grid = OccupancyGrid(maze.shape[1], maze.shape[0])

    plt.figure(figsize=(10, 5))
    plt.ion()

    for next_pose in fixed_path[1:]:
        # sense before moving
        sense_and_update(maze, robot.pose(), grid, fov_deg=90, n_beams=20, max_range=5)

        # follow the fixed path (ignores planning)
        robot.follow_path(next_pose, maze, grid=grid)
        # --- Plot ---
        plot_state(maze, robot, grid, path=fixed_path, goal=goal)
        # Stop if reached goal
        if (robot.x, robot.y) == goal:
            print("Reached goal!")
            break
    plt.ioff()
    plt.show()


def run_demo():
    maze = create_maze()
    start = (1, 8)
    goal = (8, 1)
    robot = Robot(*start, 0.0)

    # Occupancy grid
    grid = OccupancyGrid(maze.shape[1], maze.shape[0])

    plt.figure(figsize=(10, 5))
    plt.ion()

    for step in range(50):

        sense_and_update(maze, robot.pose(), grid, **LIDAR_CONFIG)
        occ_map = grid.prob_map()
        path = astar(occ_map, (robot.x, robot.y), goal, occ_threshold=0.6)
        if not path:
            print("No path found in estimated map")
            return

        # Move one step along path
        if len(path) > 1:
            next_pose = path[1]
            before = (robot.x, robot.y)
            robot.follow_path(next_pose, maze, grid=grid, **LIDAR_CONFIG)
            after = (robot.x, robot.y)
            if before == after:  # didn’t move (hit wall)
                print("Blocked! Replanning...")
                sense_and_update(maze, robot.pose(), grid, **LIDAR_CONFIG)
                return

        # --- Plot ---
        plot_state(maze, robot, grid, path=path, goal=goal)

        # Stop if reached goal
        if (robot.x, robot.y) == goal:
            print("Reached goal!")
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_demo()
    # run_demo_fixed_path()
