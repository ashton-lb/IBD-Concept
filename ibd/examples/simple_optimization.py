import numpy as np
from ibd.objective import ObjectiveFunction
from ibd.region import OptimalRegion

xs = np.linspace(-10, 10, 1000)

f = ObjectiveFunction(lambda x: -(x - 2)**2 + 5)
ys = [f.evaluate(x) for x in xs]

region = OptimalRegion(threshold=0.5)
optimal_xs = region.compute(xs, ys)

print("Optimal region:")
print(min(optimal_xs), max(optimal_xs))
