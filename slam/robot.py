import math
from slam.sensors import sense_and_update

# Discrete heading set in degrees
HEADINGS = [0, 90, 180, -90]  # E, S, W, N
DIRECTION_MAP = {
    0: (1, 0),  # east
    90: (0, 1),  # south
    180: (-1, 0),  # west
    -180: (-1, 0),  # west (wraparound)
    -90: (0, -1),  # north
}


def snap_heading(theta):
    """Snap any angle (rad) to nearest discrete heading (E,S,W,N)."""
    deg = int(round(math.degrees(theta)))
    nearest = min(HEADINGS, key=lambda h: abs(((deg - h + 180) % 360) - 180))
    return math.radians(nearest)


def heading_label(theta):
    """Return N/E/S/W string for snapped heading."""
    deg = int(round(math.degrees(snap_heading(theta))))
    mapping = {0: "E", 90: "S", 180: "W", -180: "W", -90: "N"}
    return mapping[deg]


class Robot:
    def __init__(self, x, y, theta=0.0):
        self.x = x
        self.y = y
        self.theta = snap_heading(theta)

    def pose(self):
        return (self.x, self.y, self.theta)

    def rotate(self, angle_deg):
        """Rotate robot by ±90 or 180 degrees."""
        self.theta += math.radians(angle_deg)
        self.theta = snap_heading(self.theta)
        print(f"Rotated to heading {heading_label(self.theta)}")

    def move_forward(self, maze):
        """Move one step forward if free, else stay."""
        deg = int(round(math.degrees(snap_heading(self.theta))))
        if deg not in DIRECTION_MAP:
            print(f"Invalid heading {deg}, snapping...")
            deg = min(DIRECTION_MAP.keys(), key=lambda k: abs(k - deg))

        dx, dy = DIRECTION_MAP[deg]
        nx, ny = self.x + dx, self.y + dy

        if ny < 0 or ny >= maze.shape[0] or nx < 0 or nx >= maze.shape[1]:
            print("Blocked: out of bounds")
            return

        if maze[ny, nx] == 0:
            self.x, self.y = nx, ny
            print(f"Moved to ({self.x},{self.y} looking {heading_label(self.theta)})")
        else:
            print(f"Blocked by wall at ({nx},{ny}), staying at ({self.x},{self.y})")

    def follow_path(self, next_pose, maze, grid=None, **lidar_cfg):
        (nx, ny) = next_pose
        cx, cy, ct = self.pose()

        dx, dy = nx - cx, ny - cy

        # Map desired move to heading
        if dx == 1 and dy == 0:
            desired_theta = math.radians(0)  # east
        elif dx == -1 and dy == 0:
            desired_theta = math.radians(180)  # west
        elif dx == 0 and dy == -1:
            desired_theta = math.radians(-90)  # north
        elif dx == 0 and dy == 1:
            desired_theta = math.radians(90)  # south
        else:
            print(f"Invalid step from {(cx,cy)} → {(nx,ny)}")
            return

        current = snap_heading(self.theta)
        desired = snap_heading(desired_theta)

        if current != desired:
            cur_deg = int(round(math.degrees(current)))
            des_deg = int(round(math.degrees(desired)))

            cur_idx = HEADINGS.index(cur_deg)
            des_idx = HEADINGS.index(des_deg)

            diff = (des_idx - cur_idx) % 4
            if diff == 1:  # left turn
                self.rotate(90)
            elif diff == 3:  # right turn
                self.rotate(-90)
            elif diff == 2:  # u-turn
                self.rotate(180)

            if grid is not None:
                sense_and_update(maze, self.pose(), grid, **lidar_cfg)

        # Move forward after aligning
        self.move_forward(maze)
        # print(f"At {self.pose()} heading {heading_label(self.theta)}")
