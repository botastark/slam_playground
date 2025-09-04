import heapq
import numpy as np


def heuristic(a, b):
    """Manhattan distance."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


import heapq
import numpy as np


def detect_frontiers(occ_map, free_threshold=0.3, unknown_low=0.4, unknown_high=0.6):
    """
    Detect frontier cells: free cells adjacent to unknown.
    Returns a list of (x,y).
    """
    h, w = occ_map.shape
    frontiers = []
    for y in range(h):
        for x in range(w):
            if occ_map[y, x] <= free_threshold:
                # check 4-neighbors for unknown
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if unknown_low <= occ_map[ny, nx] <= unknown_high:
                            frontiers.append((x, y))
                            break
    return frontiers


def nearest_frontier(robot, frontiers, goal=None):
    rx, ry, _ = robot.pose()

    def score(f):
        d_robot = abs(f[0] - rx) + abs(f[1] - ry)
        d_goal = abs(f[0] - goal[0]) + abs(f[1] - goal[1]) if goal else 0
        return d_robot + 0.5 * d_goal  # weighted

    return min(frontiers, key=score)


def astar(
    occupancy_grid,
    start,
    goal,
    occ_threshold=0.7,
    free_threshold=0.45,
    unknown_penalty=3,
):
    """
    Run A* on occupancy grid with special handling for unknown cells.
    - occupancy_grid: 2D array of probabilities [0=free, 1=occupied]
    - start, goal: (x,y) tuples
    - occ_threshold: cells >= this are treated as walls
    - free_threshold: cells <= this are considered free
    - unknown_penalty: extra cost added for traversing unknown cells
    """
    width, height = occupancy_grid.shape[1], occupancy_grid.shape[0]
    start, goal = tuple(start), tuple(goal)

    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = [(fscore[start], start)]

    while oheap:
        _, current = heapq.heappop(oheap)
        if current == goal:
            # reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        close_set.add(current)
        cx, cy = current

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (cx + dx, cy + dy)
            x, y = neighbor
            if x < 0 or y < 0 or x >= width or y >= height:
                continue

            p = occupancy_grid[y, x]

            # Hard wall
            if p >= occ_threshold:
                continue

            # Cost model
            step_cost = 1
            if free_threshold < p < occ_threshold:
                # Unknown cell
                step_cost += unknown_penalty

            tentative_g = gscore[current] + step_cost

            if tentative_g < gscore.get(neighbor, 1e9):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g
                fscore[neighbor] = tentative_g + heuristic(neighbor, goal)
                if neighbor not in close_set:
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return None  # no path found
