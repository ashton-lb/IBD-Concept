class ObjectiveFunction:
    """
    Wraps a mathematical objective function.
    """

    def __init__(self, func):
        self.func = func

    def evaluate(self, x):
        return self.func(x)
