import numpy as np
import matplotlib.pyplot as plt

from ibd.objective import ObjectiveFunction
from ibd.region import OptimalRegion

# Domain
xs = np.linspace(-10, 10, 1000)

# Objective function
f = ObjectiveFunction(lambda x: -(x - 2)**2 + 5)

# Evaluate
ys = [f.evaluate(x) for x in xs]
gradients = [f.derivative(x) for x in xs]

# Compute IBD region
region = OptimalRegion(
    value_threshold=0.5,
    gradient_threshold=0.05
)

optimal_xs = region.compute(xs, ys, gradients)

# Plot
plt.plot(xs, ys, label="Objective Function")
plt.scatter(
    optimal_xs,
    [f.evaluate(x) for x in optimal_xs],
    color="red",
    s=10,
    label="IBD Optimal Region"
)

plt.legend()
plt.title("IBD Concept – Region-Based Optimality")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
