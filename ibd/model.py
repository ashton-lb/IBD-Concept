import numpy as np

class IBDModel:
    """
    ML-ready interface for the IBD framework.
    """

    def __init__(self, objective, region):
        self.objective = objective
        self.region = region

    def fit(self, X):
        """
        Fit does not train weights.
        It evaluates the decision space.
        """
        self.X_ = X
        self.y_ = [self.objective.evaluate(x) for x in X]
        return self

    def predict(self):
        """
        Returns optimal region after fitting.
        """
        gradients = [self.objective.derivative(x) for x in self.X_]
        return self.region.compute(self.X_, self.y_, gradients)

    def loss(self, x):
        """
        Loss as distance from optimal region.
        """
        region = self.predict()
        if not region:
            return float("inf")
        return min(abs(x - r) for r in region)
