# import numpy as np


def create_simple_maze():
    """Ground truth maze: 1 = wall, 0 = free"""
    maze = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    return maze


import numpy as np
import random


def create_maze(width=30, height=30, complexity=0.75, density=0.75):
    """
    Generate a random maze using recursive division.
    - width, height: dimensions
    - complexity: controls how winding the paths are
    - density: controls how many walls
    """
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    maze = np.ones(shape, dtype=int)

    # Fill borders
    maze[0, :] = maze[-1, :] = 1
    maze[:, 0] = maze[:, -1] = 1
    maze[1, 1] = 0  # start point
    maze[maze.shape[0] - 2, maze.shape[1] - 2] = 0  # end point

    # Number of walls
    num_complexity = int(complexity * (shape[0] // 2) * (shape[1] // 2))
    num_density = int(density * (shape[0] // 2) * (shape[1] // 2))

    for i in range(num_density):
        x, y = (
            random.randint(0, shape[1] // 2) * 2,
            random.randint(0, shape[0] // 2) * 2,
        )
        maze[y, x] = 0
        for j in range(num_complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - 2))
            if x < shape[1] - 2:
                neighbours.append((y, x + 2))
            if y > 1:
                neighbours.append((y - 2, x))
            if y < shape[0] - 2:
                neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[random.randint(0, len(neighbours) - 1)]
                if maze[y_, x_] == 1:
                    maze[y_, x_] = 0
                    maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 0
                    x, y = x_, y_
    return maze
