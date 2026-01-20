class ObjectiveFunction:
    """
    Wraps a mathematical objective function.
    """

    def __init__(self, func):
        self.func = func

    def evaluate(self, x):
        return self.func(x)
import numpy as np

class ObjectiveFunction:
    """
    Wraps a mathematical objective function.
    """

    def __init__(self, func):
        self.func = func

    def evaluate(self, x):
        return self.func(x)

    def derivative(self, x, h=1e-5):
        """
        Numerical derivative (finite differences).
        """
        return (self.func(x + h) - self.func(x - h)) / (2 * h)
