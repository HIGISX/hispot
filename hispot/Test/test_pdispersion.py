import numpy as np
import random
from hispot.FLP import PDispersion
from pulp import *

num_points = 20
num_located = 3  # P: number of located facility in the end
np.random.seed(0)
num_people = np.random.randint(1, size=num_points)  # h
points = [(random.random(), random.random()) for i in range(num_points)]
points_np = np.array(points)
centers, obj = PDispersion(num_points=num_points,
                           points=points_np,
                           solver=PULP_CBC_CMD(),
                           num_located=num_located, ).prob_solve()
