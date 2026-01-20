class OptimalRegion:
    """
    Computes regions of acceptable optimality instead of a single optimum.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def compute(self, xs, ys):
        max_value = max(ys)
        return [
            x for x, y in zip(xs, ys)
            if y >= max_value - self.threshold
        ]
