import math
from slam.sensors import sense_and_update

# Discrete heading set in degrees
HEADINGS = [0, 90, 180, -90]  # E, S, W, N


def snap_heading(theta):
    """Snap any radian angle to the nearest of [0, 90, 180, -90] degrees."""
    deg = round(math.degrees(theta))
    # wrap into -180..180
    deg = ((deg + 180) % 360) - 180
    candidates = [0, 90, 180, -90]
    nearest = min(candidates, key=lambda h: abs(h - deg))
    return math.radians(nearest)


def heading_label(theta):
    """Return cardinal direction string (N/E/S/W) for given angle (rad)."""
    deg = int(round(math.degrees(theta)))
    mapping = {0: "E", 90: "S", 180: "W", -180: "W", -90: "N"}
    nearest = min(mapping.keys(), key=lambda k: abs(k - deg))
    return mapping[nearest]


class Robot:
    def __init__(self, x, y, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta  # radians

    def pose(self):
        return (self.x, self.y, self.theta)

    def rotate(self, angle_deg):
        """Rotate robot left/right (±90)."""
        self.theta += math.radians(angle_deg)
        # keep theta in [-pi, pi]
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

    def move_forward(self, maze):
        """Move one step forward if free, else stay."""
        dx = round(math.cos(self.theta))
        dy = round(math.sin(self.theta))
        nx, ny = self.x + dx, self.y + dy

        if ny < 0 or ny >= maze.shape[0] or nx < 0 or nx >= maze.shape[1]:
            print("Blocked: out of bounds")
            return

        if maze[ny, nx] == 0:
            self.x, self.y = nx, ny
            # print(f"Moved to ({self.x},{self.y})")
        else:
            print(f"Blocked by wall at ({nx},{ny}), staying at ({self.x},{self.y})")

    def follow_path(self, next_pose, maze, grid=None, **lidar_cfg):
        """Rotate toward next cell and move one step forward, updating map."""
        (nx, ny) = next_pose
        cx, cy, ct = self.pose()
        dx, dy = nx - cx, ny - cy

        # Map dx,dy to desired heading
        if dx == 1 and dy == 0:
            desired_theta = 0  # east
        elif dx == -1 and dy == 0:
            desired_theta = math.pi  # west
        elif dx == 0 and dy == -1:
            desired_theta = -math.pi / 2  # south
        elif dx == 0 and dy == 1:
            desired_theta = math.pi / 2  # north
        else:
            print(f"Invalid step from {(cx,cy)} → {(nx,ny)}")
            return

        # Rotate until aligned
        current = snap_heading(self.theta)
        desired = snap_heading(desired_theta)

        while current != desired:
            # figure out rotation direction on [E, S, W, N] cycle
            cur_deg = int(round(math.degrees(current)))
            des_deg = int(round(math.degrees(desired)))

            cur_idx = HEADINGS.index(cur_deg)
            des_idx = HEADINGS.index(des_deg)

            diff = (des_idx - cur_idx) % 4
            if diff == 1:  # turn left
                self.rotate(90)
            elif diff == 3:  # turn right
                self.rotate(-90)
            else:  # U-turn
                self.rotate(90)
                self.rotate(90)

            # re-snap after rotating
            current = snap_heading(self.theta)
            if grid is not None:
                sense_and_update(maze, self.pose(), grid, **lidar_cfg)

        # Now move forward
        self.move_forward(maze)
        # print(f"At {self.pose()} heading {heading_label(self.theta)}")
