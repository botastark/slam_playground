import numpy as np

class OccupancyGrid:
    def __init__(self, width, height, resolution=1.0):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.log_odds = np.zeros((height, width))
        self.L0 = 0.0   # prior log-odds (50/50)

    def prob_map(self):
        """Convert log-odds to probability map."""
        return 1 - 1 / (1 + np.exp(self.log_odds))

    def update(self, updates):
        """
        updates = list of (y, x, value) where value is +1 (occupied) or -1 (free)
        """
        for y, x, val in updates:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.log_odds[y, x] += val
