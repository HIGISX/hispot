import numpy as np
import random
from hispot.FIFLP import MiFM
from pulp import *

random.seed(1)
num_point = 10  # facility candidate sites
num_path = 5  # path num
num_m = 2  # choice facility
Vp = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
Vp_np = np.array(Vp)
points = [(random.random(), random.random()) for i in range(num_point)]  #
points_np = np.array(points)
Fp = [random.randint(5, 10) for j in range(num_path)]
Fp_np = np.array(Fp)
e = 0.3  # [0,1]                                         # 1-ϵ
Ke = (1 - e) * sum(Fp_np)  # Kϵ

selected_path, selected_vector = MiFM(num_path=num_path,
                                      num_vector=num_point,
                                      path_vector=Vp_np,
                                      path_flow=Fp_np,
                                      intercept_e=Ke,
                                      solver=GUROBI_CMD()).prob_solve()
