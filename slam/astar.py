import heapq
import numpy as np


def heuristic(a, b):
    """Manhattan distance."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(occupancy_grid, start, goal, occ_threshold=0.6):
    """
    Run A* on occupancy grid (probabilities).
    - occupancy_grid: 2D array of probabilities [0=free, 1=occupied]
    - start, goal: (x,y) tuples
    - occ_threshold: cells >= this are treated as obstacles
    """
    width, height = occupancy_grid.shape[1], occupancy_grid.shape[0]
    start, goal = tuple(start), tuple(goal)
    # print("Start cell prob:", occupancy_grid[start[1], start[0]])
    # print("Goal cell prob:", occupancy_grid[goal[1], goal[0]])

    # closed set & open list
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
            if occupancy_grid[y, x] >= occ_threshold:  # treat as blocked
                # print(f"Blocked cell {neighbor} prob={occupancy_grid[y,x]:.2f}")
                continue
            # else:
            # print(f"Free cell {neighbor} prob={occupancy_grid[y,x]:.2f}")

            tentative_g = gscore[current] + 1
            if tentative_g < gscore.get(neighbor, 1e9):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g
                fscore[neighbor] = tentative_g + heuristic(neighbor, goal)
                if neighbor not in close_set:
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return None  # no path found
