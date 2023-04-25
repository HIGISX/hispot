import numpy as np
import random
from hispot.FIFLP import MaAG
from pulp import *

num_point = 10  # facility candidate sites
num_path = 5  # path num
num_m = 2  # choice facility
Vp = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
Vp_np = np.array(Vp)  # Vp:Path
points = [(random.random(), random.random()) for i in range(num_point)]  # V
points_np = np.array(points)
api = np.random.randint(10, size=(num_path, num_point))  # api
selected_path, selected_vector = MaAG(num_path=num_path,
                                      num_vector=num_point,
                                      num_choice=num_m,
                                      path_vector=Vp_np,
                                      vector_gain=api,
                                      solver=GUROBI_CMD()).prob_solve()
