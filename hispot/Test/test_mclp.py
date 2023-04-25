import numpy as np
import random
from hispot.coverage import MCLP
from pulp import *

n_points = 20
p = 4
radius = 0.2
num_people = np.random.randint(1, 2, size=n_points)
points = [(random.random(), random.random()) for i in range(n_points)]
points_np = np.array(points)

centers, obj = MCLP(num_points=n_points,
                    num_located=p,
                    num_people=num_people,
                    points=points_np,
                    radius=radius,
                    solver=PULP_CBC_CMD()).prob_solve()
