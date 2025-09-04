class Robot:
    def __init__(self, x, y, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta  # radians

    def pose(self):
        return (self.x, self.y, self.theta)

    def move(self, dx, dy, dtheta=0.0):
        self.x += dx
        self.y += dy
        self.theta += dtheta
