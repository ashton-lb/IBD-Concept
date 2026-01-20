import numpy as np
import matplotlib.pyplot as plt

from ibd.objective2d import ObjectiveFunction2D
from ibd.region2d import OptimalRegion2D

# Grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Objective
f = ObjectiveFunction2D(lambda x, y: -(x - 1)**2 - (y + 2)**2 + 10)

values = []
gradients = []
points_x = []
points_y = []

for i in range(len(x)):
    for j in range(len(y)):
        xv, yv = X[i, j], Y[i, j]
        points_x.append(xv)
        points_y.append(yv)
        values.append(f.evaluate(xv, yv))
        gradients.append(f.gradient(xv, yv))

region = OptimalRegion2D(
    value_threshold=0.5,
    gradient_norm_threshold=0.2
)

optimal_points = region.compute(points_x, points_y, values, gradients)

# Plot
plt.scatter(points_x, points_y, c=values, cmap="viridis", s=5)
if optimal_points:
    ox, oy = zip(*optimal_points)
    plt.scatter(ox, oy, color="red", s=10)

plt.title("IBD Concept – 2D Optimal Region")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="f(x, y)")
plt.show()
