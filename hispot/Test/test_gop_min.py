import numpy as np
import random
from hispot.FIFLP import MiFMG
from pulp import *

num_point = 10  # facility candidate sites
num_path = 5  # path num
Vp = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
Vp_np = np.array(Vp)  # Vp:Path
Vp = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])  # Vp:Path
points = [(random.random(), random.random()) for i in range(num_point)]  # V
points_np = np.array(points)
api = np.random.randint(10, size=(num_path, num_point))
e = 0.4  # 1-ϵ
G = np.amax(api, axis=1).sum()  # G
Ge = (1 - e) * G  # Gϵ = (1-ϵ)G
selected_path, selected_vector = MiFMG(num_path=num_path,
                                       num_vector=num_point,
                                       path_vector=Vp_np,
                                       vector_gain=api,
                                       flow_gain=Ge,
                                       solver=GUROBI_CMD()).prob_solve()
