import matplotlib.pyplot as plt
from slam.gridmap import OccupancyGrid
from slam.sensors import inverse_sensor_model, sense_and_update, cast_ray
from slam.robot import Robot
from slam.maze import create_maze
from slam.astar import astar
import numpy as np
import math

def heading_label(theta):
    """Return N/E/S/W string for given angle in radians."""
    deg = int(round(math.degrees(theta)))
    mapping = {0:"E", 90:"S", 180:"W", -180:"W", -90:"N"}
    nearest = min(mapping.keys(), key=lambda k: abs(k-deg))
    return mapping[nearest]

def follow_path(robot, next_pose, maze):
    (nx, ny) = next_pose
    cx, cy, ct = robot.pose()

    dx, dy = nx - cx, ny - cy

    # Map dx,dy to desired heading
    if dx == 1 and dy == 0:   desired_theta = 0            # east
    elif dx == -1 and dy == 0: desired_theta = math.pi     # west
    elif dx == 0 and dy == -1: desired_theta = -math.pi/2  # south
    elif dx == 0 and dy == 1:  desired_theta = math.pi/2   # north
    else:
        print(f"Invalid step from {(cx,cy)} → {(nx,ny)}")
        return
    # Rotate in 90° increments until heading matches
    while round(robot.theta, 3) != round(desired_theta, 3):
        # difference in degrees
        diff = (desired_theta - robot.theta + math.pi) % (2*math.pi) - math.pi
        if diff > 0:
            robot.rotate(90)
        else:
            robot.rotate(-90)

    # Now move forward
    robot.move_forward(maze)
    print(f"At {robot.pose()} heading {heading_label(robot.theta)}")

def run_demo():
    maze = create_maze()
    start = (1, 8)
    goal = (8, 1)
    robot = Robot(*start, 0.0)
   # Run A*
    occ_grid = OccupancyGrid(maze.shape[1], maze.shape[0])
    path = astar(maze, start, goal, occ_threshold=0.5)  # you may need to adapt astar to use maze directly
    if not path:
        print("No path found!")
        return
    plt.figure()
    plt.ion()


    for next_pose in path[1:]:
        follow_path(robot, next_pose, maze)

        # Draw
        plt.imshow(maze, cmap="gray_r", origin="upper")
        plt.plot(robot.x, robot.y, "ro")

        # arrow
        plt.arrow(robot.x, robot.y,
                  0.5*math.cos(robot.theta),
                  0.5*math.sin(robot.theta),
                  head_width=0.2, head_length=0.2, fc='r', ec='r')

        # label
        plt.text(robot.x + 0.3, robot.y + 0.3, heading_label(robot.theta),
                 color="red", fontsize=12, weight="bold")

        # path
        px, py = zip(*path)
        plt.plot(px, py, "b.-")

        plt.pause(0.5)
        plt.clf()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_demo()