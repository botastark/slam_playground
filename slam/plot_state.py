import matplotlib.pyplot as plt
import numpy as np
import math
from slam.sensors import cast_ray

LIDAR_CONFIG = {"fov_deg": 90, "n_beams": 30, "max_range": 5}


def plot_state(maze, robot, grid, path=None, goal=None, lidar_cfg=None):
    if lidar_cfg is None:
        lidar_cfg = LIDAR_CONFIG
    fov = math.radians(lidar_cfg["fov_deg"])
    rel_angles = np.linspace(-fov / 2, fov / 2, lidar_cfg["n_beams"])
    max_range = lidar_cfg["max_range"]

    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(maze, cmap="gray_r", origin="upper")
    plt.plot(robot.x, robot.y, "ro")
    plt.arrow(
        robot.x,
        robot.y,
        0.5 * math.cos(robot.theta),
        0.5 * math.sin(robot.theta),
        head_width=0.2,
        head_length=0.2,
        fc="r",
        ec="r",
    )

    # planned path
    if path is not None:
        px, py = zip(*path)
        plt.plot(px, py, "b.-")
    if goal is not None:
        plt.plot(goal[0], goal[1], "gx")

    # lidar rays

    plt.title("Ground Truth")

    # --- occupancy grid ---
    plt.subplot(1, 2, 2)
    plt.imshow(1 - grid.prob_map(), cmap="gray", origin="upper", vmin=0, vmax=1)
    plt.plot(robot.x, robot.y, "ro")
    if path is not None:
        px, py = zip(*path)
        plt.plot(px, py, "b.-")
    if goal is not None:
        plt.plot(goal[0], goal[1], "gx")

    rx, ry, rt = robot.pose()
    for rel in rel_angles:
        ang = rt + rel
        free_cells, occ_cell = cast_ray(maze, robot.pose(), ang, max_range=max_range)
        if occ_cell is not None:
            plt.plot([rx, occ_cell[0]], [ry, occ_cell[1]], "g--", alpha=0.5)
        elif free_cells:
            fx, fy = free_cells[-1]
            plt.plot([rx, fx], [ry, fy], "g--", alpha=0.3)

    plt.title("Estimated Occupancy Map")

    plt.pause(1)
