import math
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
        self.theta = (self.theta + math.pi) % (2*math.pi) - math.pi

    def move_forward(self, maze):
        """Move one step forward if free, else stay."""
        dx = round(math.cos(self.theta))
        dy = round(math.sin(self.theta))

        nx = self.x + dx
        ny = self.y + dy   # minus because maze row increases downward

        if ny < 0 or ny >= maze.shape[0] or nx < 0 or nx >= maze.shape[1]:
            print("Blocked: out of bounds")
            return

        if maze[ny, nx] == 0:
            self.x, self.y = nx, ny
        else:
            print(f"Blocked by wall at ({nx},{ny}), staying at ({self.x},{self.y})")

    def follow_path(self, next_pose, maze):
        (nx, ny) = next_pose
        cx, cy, ct = self.pose()

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
        while round(self.theta, 3) != round(desired_theta, 3):
            # difference in degrees
            diff = (desired_theta - self.theta + math.pi) % (2*math.pi) - math.pi
            if diff > 0:
                self.rotate(90)
            else:
                self.rotate(-90)

        # Now move forward
        self.move_forward(maze)
        # print(f"At {self.pose()} heading {heading_label(self.theta)}")

def heading_label(theta):
    """Return N/E/S/W string for given angle in radians."""
    deg = int(round(math.degrees(theta)))
    mapping = {0:"E", 90:"S", 180:"W", -180:"W", -90:"N"}
    nearest = min(mapping.keys(), key=lambda k: abs(k-deg))
    return mapping[nearest]