import math
import numpy as np


def bresenham(x0, y0, x1, y1):
    """Grid cells from (x0,y0) to (x1,y1)."""
    points = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def cast_ray(maze, robot_pose, angle, max_range=10, step_size=0.1):
    x, y, theta = robot_pose
    dx, dy = math.cos(angle), math.sin(angle)

    free_cells = []
    r = 0.0
    while r < max_range:
        r += step_size
        xi = int(round(x + r * dx))
        yi = int(round(y + r * dy))

        if yi < 0 or yi >= maze.shape[0] or xi < 0 or xi >= maze.shape[1]:
            return free_cells, None
        if maze[yi, xi] == 1:
            return free_cells, (xi, yi)
        free_cells.append((xi, yi))
    return free_cells, None


def inverse_sensor_model(robot, endpoint):
    (x0, y0, _) = robot
    (x1, y1) = endpoint
    pts = bresenham(int(x0), int(y0), x1, y1)

    updates = []
    # Mark free cells until but not including the last
    for x, y in pts[:-1]:
        updates.append((y, x, -1))  # free
    # Last cell = occupied (wall hit)
    updates.append((y1, x1, +1))
    return updates


def sense_and_update(maze, robot, grid, **LIDAR_CONFIG):
    fov_deg = LIDAR_CONFIG.get("fov_deg", 90)
    n_beams = LIDAR_CONFIG.get("n_beams", 15)
    max_range = LIDAR_CONFIG.get("max_range", 10)

    x, y, theta = robot

    fov = math.radians(fov_deg)
    rel_angles = np.linspace(-fov / 2, fov / 2, n_beams)

    for rel in rel_angles:
        ang = theta + rel
        free_cells, occ_cell = cast_ray(maze, (x, y, theta), ang, max_range)

        for xi, yi in free_cells:
            grid.log_odds[yi, xi] += -1
        if occ_cell is not None:
            xi, yi = occ_cell
            grid.log_odds[yi, xi] += +1
