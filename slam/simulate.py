import matplotlib.pyplot as plt
from slam.gridmap import OccupancyGrid
from slam.sensors import inverse_sensor_model, sense_and_update, cast_ray
from slam.robot import Robot, heading_label
from slam.maze import create_maze, create_simple_maze
from slam.astar import astar, detect_frontiers, nearest_frontier
from slam.plot_state import plot_state
import numpy as np
import math

LIDAR_CONFIG = {"fov_deg": 90, "n_beams": 30, "max_range": 5}
N_COMMIT = 5


def run_demo_fixed_path(maze, robot, goal, grid, LIDAR_CONFIG):
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

    plt.figure(figsize=(10, 5))
    plt.ion()

    for next_pose in fixed_path[1:]:
        # sense before moving
        sense_and_update(maze, robot.pose(), grid, **LIDAR_CONFIG)
        # follow the fixed path (ignores planning)
        robot.follow_path(next_pose, maze, grid=grid, **LIDAR_CONFIG)
        # --- Plot ---
        plot_state(maze, robot, grid, path=fixed_path, goal=goal)
        # Stop if reached goal
        if (robot.x, robot.y) == goal:
            print("Reached goal!")
            break
    plt.ioff()
    plt.show()


def run_demo(maze, robot, goal, grid, LIDAR_CONFIG):

    plt.figure(figsize=(10, 5))
    plt.ion()
    plot_state(maze, robot, grid, goal=goal)
    current_frontier = None
    commit_steps = 0
    for step in range(100):

        sense_and_update(maze, robot.pose(), grid, **LIDAR_CONFIG)
        occ_map = grid.prob_map()
        path = astar(occ_map, (robot.x, robot.y), goal, occ_threshold=0.6)
        # --- Frontier fallback if path is missing or trivial (<=2) ---
        if not path or len(path) <= 2:
            # --- Fallback: frontier exploration ---
            print("No path to goal, switching to frontier exploration...")
            frontiers = detect_frontiers(occ_map)
            if not frontiers:
                print("No frontiers left — map fully explored, goal truly unreachable.")
                break
            if current_frontier is None or commit_steps <= 0:
                # Pick new frontier
                current_frontier = nearest_frontier(robot, frontiers, goal=goal)
                commit_steps = N_COMMIT
                print(f"Picked new frontier {current_frontier}")
            # Plan toward current frontier
            path = astar(
                occ_map, (robot.x, robot.y), current_frontier, occ_threshold=0.7
            )
            if not path:
                print(f"Frontier {current_frontier} unreachable, dropping it")
                current_frontier = None
                continue
            else:
                commit_steps -= 1
                print(
                    f"Heading to frontier {current_frontier}, commit_steps={commit_steps}"
                )

        # Move one step along path
        if len(path) > 1:
            print("Planned path:", path[:10], "...")
            next_pose = path[1]
            robot.follow_path(next_pose, maze, grid=grid, **LIDAR_CONFIG)

        # --- Plot ---
        plot_state(
            maze,
            robot,
            grid,
            path=path,
            goal=goal,
            frontiers=[current_frontier] if current_frontier is not None else [],
        )

        # Stop if reached goal
        if (robot.x, robot.y) == goal:
            print("Reached goal!")
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # maze = create_simple_maze()
    maze = create_maze(10, 9, complexity=0.6, density=0.6)

    start = (1, 1)
    goal = (maze.shape[1] - 2, maze.shape[0] - 2)
    robot = Robot(*start, 0.0)
    assert maze[start[1], start[0]] == 0
    assert maze[goal[1], goal[0]] == 0

    # Occupancy grid
    print(f"Maze:\n{maze}")
    grid = OccupancyGrid(maze.shape[1], maze.shape[0])
    run_demo(maze, robot, goal, grid, LIDAR_CONFIG)
    # run_demo_fixed_path()
