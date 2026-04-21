import matplotlib.pyplot as plt
from slam.gridmap import OccupancyGrid
from slam.sensors import inverse_sensor_model, sense_and_update, cast_ray
from slam.robot import Robot, heading_label
from slam.maze import create_maze, create_simple_maze, resample_maze
from slam.astar import astar, detect_frontiers, nearest_frontier
from slam.plot_state import plot_state
import numpy as np
import math

# === UNIT SYSTEM ===
# Physical room dimensions in meters
ROOM_WIDTH_M = 10.0
ROOM_HEIGHT_M = 10.0

# Grid resolution: cells per meter (e.g., 10 = 10cm resolution)
CELLS_PER_METER = 10

# Sensor configuration in meters
SENSOR_RANGE_M = 5.0
SENSOR_FOV_DEG = 90
SENSOR_BEAMS = 30

# Derived configuration
GRID_GRANULARITY = CELLS_PER_METER
LIDAR_CONFIG = {
    "fov_deg": SENSOR_FOV_DEG,
    "n_beams": SENSOR_BEAMS,
    "max_range": SENSOR_RANGE_M * CELLS_PER_METER,  # convert m to cells
    "step_size": 5 * 1.0 / CELLS_PER_METER,  # step in cells (= 0.1m for 10 cells/m)
}
N_COMMIT = 5


def find_free_cells(maze, margin=5):
    """Find all free cells in the maze with some margin from walls."""
    free_cells = []
    h, w = maze.shape
    for y in range(margin, h - margin):
        for x in range(margin, w - margin):
            if maze[y, x] == 0:
                free_cells.append((x, y))
    return free_cells


def select_start_and_goal(maze, min_distance=None):
    """
    Select valid start and goal positions from the maze.
    Returns (start, goal) as (x, y) tuples.
    """
    free_cells = find_free_cells(maze, margin=GRID_GRANULARITY)

    if len(free_cells) < 2:
        raise ValueError("Not enough free cells in maze to select start and goal")

    # If no minimum distance specified, use half the diagonal
    if min_distance is None:
        min_distance = int(0.5 * np.sqrt(maze.shape[0] ** 2 + maze.shape[1] ** 2))

    # Try to find start near top-left area
    start_candidates = [
        (x, y)
        for x, y in free_cells
        if x < maze.shape[1] // 3 and y < maze.shape[0] // 3
    ]
    if not start_candidates:
        start_candidates = free_cells[: len(free_cells) // 3]

    # Select start from candidates
    start = (
        start_candidates[len(start_candidates) // 2]
        if start_candidates
        else free_cells[0]
    )

    # Find goal far from start (preferably bottom-right area)
    goal_candidates = [
        (x, y)
        for x, y in free_cells
        if np.sqrt((x - start[0]) ** 2 + (y - start[1]) ** 2) >= min_distance
        and x > 2 * maze.shape[1] // 3
        and y > 2 * maze.shape[0] // 3
    ]

    if not goal_candidates:
        # Fallback: just find farthest point
        goal_candidates = [
            (x, y)
            for x, y in free_cells
            if np.sqrt((x - start[0]) ** 2 + (y - start[1]) ** 2) >= min_distance
        ]

    if not goal_candidates:
        # Last resort: use farthest available point
        goal = max(
            free_cells,
            key=lambda p: np.sqrt((p[0] - start[0]) ** 2 + (p[1] - start[1]) ** 2),
        )
    else:
        goal = goal_candidates[0]

    print(f"Selected start: {start}, goal: {goal}")
    print(
        f"Distance: {np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2):.1f} cells"
    )

    return start, goal


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


def run_demo(maze, robot, goal, grid, LIDAR_CONFIG, exploration_mode=False):

    plt.figure(figsize=(10, 5))
    plt.ion()

    if exploration_mode:
        initial_msg = "Exploration mode - mapping environment..."
    else:
        initial_msg = "Starting exploration..."

    plot_state(
        maze,
        robot,
        grid,
        goal=goal,
        lidar_cfg=LIDAR_CONFIG,
        status_msg=initial_msg,
    )
    current_frontier = None
    commit_steps = 0
    step = 0
    status_message = "Exploring..."
    unreachable_frontiers = set()  # Track frontiers we've tried and failed to reach
    while True:
        step += 1
        print(f"\n=== Step {step} === Robot at ({robot.x}, {robot.y})")

        sense_and_update(maze, robot.pose(), grid, **LIDAR_CONFIG)
        occ_map = grid.prob_map()

        # In exploration mode, skip goal-seeking and go straight to frontier exploration
        if not exploration_mode:
            path = astar(occ_map, (robot.x, robot.y), goal, occ_threshold=0.6)
        else:
            path = None

        # --- Frontier fallback if path is missing or trivial (<=2) ---
        if not path or len(path) <= 2:
            # --- Fallback: frontier exploration ---
            if exploration_mode:
                print("Continuing frontier-based exploration...")
                status_message = "Exploring environment - mapping frontiers..."
            else:
                print("No path to goal, switching to frontier exploration...")
                status_message = "No direct path - exploring frontiers..."

            frontiers = detect_frontiers(occ_map)

            # Filter out unreachable frontiers and current position
            available_frontiers = [
                f
                for f in frontiers
                if f not in unreachable_frontiers
                and f != (robot.x, robot.y)  # Don't select current position
            ]

            if not available_frontiers:
                if exploration_mode:
                    print("No frontiers left — exploration complete!")
                    status_message = "✓ EXPLORATION COMPLETE - All areas mapped"
                else:
                    print("No reachable frontiers left — goal is unreachable.")
                    status_message = (
                        "❌ GOAL UNREACHABLE - All frontiers explored, no path exists"
                    )
                break
            if current_frontier is None or commit_steps <= 0:
                # Pick new frontier - prioritize closer ones for exploration efficiency
                if exploration_mode:
                    # In exploration mode, pick nearest frontier to minimize travel
                    current_frontier = min(
                        available_frontiers,
                        key=lambda f: abs(f[0] - robot.x) + abs(f[1] - robot.y),
                    )
                else:
                    # In navigation mode, bias toward goal
                    current_frontier = nearest_frontier(
                        robot, available_frontiers, goal=goal
                    )

                commit_steps = N_COMMIT
                print(f"Picked new frontier {current_frontier}")
                status_message = f"Exploring new frontier at {current_frontier}"
            # Plan toward current frontier
            path = astar(
                occ_map, (robot.x, robot.y), current_frontier, occ_threshold=0.7
            )
            if not path or len(path) <= 1:  # Changed from just checking None
                print(f"Frontier {current_frontier} unreachable or at current position, dropping it")
                unreachable_frontiers.add(current_frontier)
                current_frontier = None
                status_message = "Frontier unreachable - searching for new target..."
                continue
            else:
                commit_steps -= 1
                print(
                    f"Heading to frontier {current_frontier}, commit_steps={commit_steps}"
                )
                status_message = f"Moving to frontier (steps left: {commit_steps})"
        else:
            status_message = f"Following path to goal ({len(path)} steps remaining)"

        # Move one step along path
        if len(path) > 1:
            print("Planned path:", path[:10], "...")
            next_pose = path[1]
            robot.follow_path(next_pose, maze, grid=grid, **LIDAR_CONFIG)
        else:
            print(f"Path too short ({len(path) if path else 0}), cannot move")

        # --- Plot ---
        try:
            plot_state(
                maze,
                robot,
                grid,
                path=path,
                goal=goal,
                frontiers=[current_frontier] if current_frontier is not None else [],
                lidar_cfg=LIDAR_CONFIG,
                status_msg=status_message,
            )
        except Exception as e:
            print(f"Plot error: {e}")
            break

        # Stop if reached goal
        if not exploration_mode and (robot.x, robot.y) == goal:
            print("Reached goal!")
            status_message = "✓ GOAL REACHED!"
            plot_state(
                maze,
                robot,
                grid,
                path=path,
                goal=goal,
                frontiers=[],
                lidar_cfg=LIDAR_CONFIG,
                status_msg=status_message,
            )
            break

    print(f"\nSimulation ended at step {step}, robot at ({robot.x}, {robot.y})")

    # Show final state for unreachable goal or completed exploration
    if exploration_mode or (robot.x, robot.y) != goal:
        plot_state(
            maze,
            robot,
            grid,
            path=None,
            goal=goal if not exploration_mode else None,
            frontiers=[],
            lidar_cfg=LIDAR_CONFIG,
            status_msg=status_message,
        )

    plt.ioff()
    plt.show()


def get_user_mode_selection():
    """Get user's choice of operation mode."""
    print("\n" + "=" * 50)
    print("SLAM Playground - Mode Selection")
    print("=" * 50)
    print("1. Exploration Mode - Map the environment (no specific goal)")
    print("2. Navigation Mode - Navigate to a specific goal position")
    print("=" * 50)

    while True:
        try:
            choice = input("Select mode (1 or 2): ").strip()
            if choice in ["1", "2"]:
                return int(choice)
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)
        except:
            print("Invalid input. Please enter 1 or 2.")


def get_goal_position(maze):
    """Get goal position from user with validation."""
    print("\n" + "=" * 50)
    print(f"Maze dimensions: {maze.shape[1]} x {maze.shape[0]} cells")
    print(
        f"Room size: {maze.shape[1] * (1.0/CELLS_PER_METER):.1f}m x {maze.shape[0] * (1.0/CELLS_PER_METER):.1f}m"
    )
    print("=" * 50)

    while True:
        try:
            x = input(f"Enter goal X position (0-{maze.shape[1]-1}): ").strip()
            y = input(f"Enter goal Y position (0-{maze.shape[0]-1}): ").strip()

            x, y = int(x), int(y)

            if x < 0 or x >= maze.shape[1] or y < 0 or y >= maze.shape[0]:
                print(
                    f"Position out of bounds! Must be within (0-{maze.shape[1]-1}, 0-{maze.shape[0]-1})"
                )
                continue

            if maze[y, x] != 0:
                print(f"Position ({x}, {y}) is blocked by a wall! Choose a free cell.")
                continue

            print(f"Goal set to ({x}, {y})")
            return (x, y)

        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)
        except ValueError:
            print("Invalid input. Please enter integer values.")
        except:
            print("Error processing input. Please try again.")


if __name__ == "__main__":
    # Create base maze in meters, then upsample to desired cell resolution
    # Base maze dimensions represent room structure in meters
    base_maze = create_maze(
        int(ROOM_WIDTH_M), int(ROOM_HEIGHT_M), complexity=0.6, density=0.6
    )
    maze = resample_maze(base_maze, granularity=GRID_GRANULARITY)

    # Get user mode selection
    mode = get_user_mode_selection()

    # Select start position
    if mode == 1:
        # Exploration mode - use default start
        start, _ = select_start_and_goal(maze)
        goal = None
        exploration_mode = True
        print(f"\nExploration Mode - Starting at {start}")
    else:
        # Navigation mode - get custom goal
        start, auto_goal = select_start_and_goal(maze)
        print(f"\nNavigation Mode - Starting at {start}")
        print(f"Auto-suggested goal: {auto_goal}")

        use_custom = input("Use custom goal? (y/n, default=n): ").strip().lower()
        if use_custom == "y":
            goal = get_goal_position(maze)
        else:
            goal = auto_goal
            print(f"Using auto-suggested goal: {goal}")

        exploration_mode = False

    robot = Robot(*start, 0.0)
    assert maze[start[1], start[0]] == 0, f"Start position {start} is blocked"
    if goal is not None:
        assert maze[goal[1], goal[0]] == 0, f"Goal position {goal} is blocked"

    # Occupancy grid with resolution in meters/cell
    print(f"\nRoom: {ROOM_WIDTH_M}m x {ROOM_HEIGHT_M}m")
    print(f"Grid: {maze.shape[1]} x {maze.shape[0]} cells ({CELLS_PER_METER} cells/m)")
    print(f"Sensor range: {SENSOR_RANGE_M}m")
    print(f"Mode: {'Exploration' if exploration_mode else 'Navigation'}")
    if goal:
        print(f"Goal: {goal}")
    print("\nStarting simulation...\n")

    grid = OccupancyGrid(maze.shape[1], maze.shape[0], resolution=1.0 / CELLS_PER_METER)
    run_demo(maze, robot, goal, grid, LIDAR_CONFIG, exploration_mode=exploration_mode)
    # run_demo_fixed_path()
