import numpy as np

class RobustEvaluator:
    """
    Evaluates objective robustness under noise.
    """

    def __init__(self, objective, noise_std, samples=50):
        self.objective = objective
        self.noise_std = noise_std
        self.samples = samples

    def evaluate(self, x):
        values = []
        for _ in range(self.samples):
            noise = np.random.normal(0, self.noise_std)
            values.append(self.objective.evaluate(x + noise))
        return np.mean(values)
