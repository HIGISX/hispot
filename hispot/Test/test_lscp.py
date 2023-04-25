import numpy as np
import random
from hispot.coverage import LSCP
from pulp import *

n_points = 20
radius = 0.2
points = [(random.random(), random.random()) for i in range(n_points)]
points_np = np.array(points)

centers, obj = LSCP(num_points=n_points,
                    points=points_np,
                    solver=PULP_CBC_CMD(),
                    radius=radius).prob_solve()
