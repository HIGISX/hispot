import numpy as np
import random
from hispot.coverage import MEXCLP
from pulp import *

n_points = 20
prob = 0.2
p = 4
radius = 0.2
demand = np.random.randint(1, 2, size=n_points)
points = [(random.random(), random.random()) for i in range(n_points)]
points_np = np.array(points)

centers, obj = MEXCLP(num_located=p,
                      demand=demand,
                      num_points=n_points,
                      points=points_np,
                      radius=radius,
                      unprob_rate=prob,
                      solver=PULP_CBC_CMD()).prob_solve()
