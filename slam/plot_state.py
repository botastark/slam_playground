import matplotlib.pyplot as plt
import numpy as np
import math
from slam.sensors import cast_ray

LIDAR_CONFIG = {"fov_deg": 90, "n_beams": 30, "max_range": 5}


def plot_state(
    maze,
    robot,
    grid,
    path=None,
    goal=None,
    frontiers=None,
    lidar_cfg=None,
    status_msg=None,
):
    if lidar_cfg is None:
        lidar_cfg = LIDAR_CONFIG
    fov = math.radians(lidar_cfg["fov_deg"])
    rel_angles = np.linspace(-fov / 2, fov / 2, lidar_cfg["n_beams"])
    max_range = lidar_cfg["max_range"]
    step_size = lidar_cfg.get("step_size", 1.0)

    # Calculate display metrics
    cells_per_meter = 1.0 / grid.resolution if grid.resolution > 0 else 1
    sensor_range_m = max_range * grid.resolution
    room_width_m = maze.shape[1] * grid.resolution
    room_height_m = maze.shape[0] * grid.resolution

    plt.clf()

    # Create title with configuration info
    config_text = (
        f"Room: {room_width_m:.1f}m × {room_height_m:.1f}m  |  "
        f"Grid: {maze.shape[1]}×{maze.shape[0]} cells ({cells_per_meter:.0f} cells/m)  |  "
        f"Sensor: {sensor_range_m:.1f}m range, {lidar_cfg['fov_deg']}° FOV, {lidar_cfg['n_beams']} beams"
    )
    plt.suptitle(config_text, fontsize=10, y=0.98)

    # Add status message if provided
    if status_msg:
        plt.figtext(
            0.5,
            0.02,
            status_msg,
            ha="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
            weight="bold",
        )

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

    # Frontiers
    if frontiers:
        fx, fy = zip(*frontiers)
        plt.scatter(fx, fy, c="orange", s=100, marker="o", label="Frontiers")
    plt.legend(loc="upper right")
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
        free_cells, occ_cell = cast_ray(
            maze, robot.pose(), ang, max_range=max_range, step_size=step_size
        )
        if occ_cell is not None:
            plt.plot([rx, occ_cell[0]], [ry, occ_cell[1]], "g--", alpha=0.5)
        elif free_cells:
            fx, fy = free_cells[-1]
            plt.plot([rx, fx], [ry, fy], "g--", alpha=0.3)

    plt.title("Estimated Occupancy Map")

    plt.pause(0.5)
