import numpy as np

class ObjectiveFunction2D:
    """
    2D objective function f(x, y).
    """

    def __init__(self, func):
        self.func = func

    def evaluate(self, x, y):
        return self.func(x, y)

    def gradient(self, x, y, h=1e-5):
        """
        Numerical gradient (finite differences).
        """
        df_dx = (self.func(x + h, y) - self.func(x - h, y)) / (2 * h)
        df_dy = (self.func(x, y + h) - self.func(x, y - h)) / (2 * h)
        return np.array([df_dx, df_dy])
