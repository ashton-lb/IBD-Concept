import numpy as np

class OptimalRegion2D:
    """
    Region-based optimality in R^2.
    """

    def __init__(self, value_threshold, gradient_norm_threshold):
        self.value_threshold = value_threshold
        self.gradient_norm_threshold = gradient_norm_threshold

    def compute(self, xs, ys, values, gradients):
        max_value = np.max(values)
        region = []

        for x, y, v, g in zip(xs, ys, values, gradients):
            if (
                v >= max_value - self.value_threshold
                and np.linalg.norm(g) <= self.gradient_norm_threshold
            ):
                region.append((x, y))

        return region
