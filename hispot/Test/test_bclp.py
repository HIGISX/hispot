import numpy as np
import random
from hispot.coverage import BCLP
from pulp import *

n_points = 20
p = 4
radius = 0.2

setup_cost = np.random.randint(1, 2, size=n_points)
points = [(random.random(), random.random()) for i in range(n_points)]
points_np = np.array(points)

centers, cover_twice, obj = BCLP(num_located=p,
                                 num_points=n_points,
                                 points=points_np,
                                 radius=radius,
                                 setup_cost=setup_cost,
                                 solver=PULP_CBC_CMD()).prob_solve()
