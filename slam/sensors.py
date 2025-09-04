import math
import numpy as np

def bresenham(x0, y0, x1, y1):
    """Grid cells from (x0,y0) to (x1,y1)."""
    points = []
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2*err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def cast_ray(maze, robot, angle, max_range=10):
    """Cast a ray at given angle until hitting wall or max range."""
    x, y, theta = robot
    dx, dy = math.cos(theta+angle), math.sin(theta+angle)
    for r in range(1, max_range+1):
        xi, yi = int(round(x + r*dx)), int(round(y + r*dy))
        if maze[yi, xi] == 1:   # hit wall
            return (xi, yi)
    return (xi, yi)

def inverse_sensor_model(robot, endpoint):
    """
    Ray-casting: free cells along the ray, occupied at endpoint.
    Returns updates = [(y,x,val), ...]
    """
    (x0, y0, _) = robot
    (x1, y1) = endpoint
    points = bresenham(x0, y0, x1, y1)

    updates = [(y, x, -1) for (x, y) in points[:-1]]  # free
    updates.append((y1, x1, +1))                      # occupied
    return updates

def sense_and_update(maze, robot, log_odds, fov_deg=90, n_beams=9, max_range=10):
    """Simulate lidar with limited field of view."""
    fov = math.radians(fov_deg)
    angles = [a for a in 
              list(np.linspace(-fov/2, fov/2, n_beams))]
    for ang in angles:
        endpoint = cast_ray(maze, robot, ang, max_range)
        for (y, x, val) in inverse_sensor_model(robot, endpoint):
            log_odds[y, x] += val
    return log_odds