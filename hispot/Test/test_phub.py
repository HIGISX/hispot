import numpy as np
import random
from hispot.FLP import PHub
from pulp import *

# random.seed(1)
num_points = 50
num_hubs = 10
PC, PT, PD = 1.0, 0.75, 1.25
weight = np.random.randint(1, 100, size=(num_points, num_points))
points = [(random.random() * 100, random.random() * 100) for i in range(num_points)]
points_np = np.array(points)

hubs, assigns, obj = PHub(num_points=num_points,
                          points=points_np,
                          solver=PULP_CBC_CMD(),
                          num_located=num_hubs,
                          weight=weight,
                          collect_cost=PC,
                          transfer_cost=PT,
                          distribution_cost=PD).prob_solve()
