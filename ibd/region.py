class OptimalRegion:
    """
    Computes regions of stability based on objective value
    and near-zero derivative.
    """

    def __init__(self, value_threshold, gradient_threshold):
        self.value_threshold = value_threshold
        self.gradient_threshold = gradient_threshold

    def compute(self, xs, ys, gradients):
        max_value = max(ys)

        region = []
        for x, y, g in zip(xs, ys, gradients):
            if (
                y >= max_value - self.value_threshold
                and abs(g) <= self.gradient_threshold
            ):
                region.append(x)

        return region
