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