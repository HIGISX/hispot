import numpy as np
import random
from hispot.LRP import LRP_cap
from pulp import *

random.seed(1)
num_fa = 5
num_de = 20
facility_nodes = [(random.random(), random.random()) for i in range(num_fa)]
fa_cap = [(random.randint(30, 35), random.randint(35, 40)) for i in range(num_fa)]
demand_nodes = [(random.random(), random.random()) for i in range(num_de)]
de_demand = [random.randint(1, 10) for i in range(num_de)]

facility_nodes_np = np.array(facility_nodes)
demand_nodes_np = np.array(demand_nodes)

selected, assigned, obj = LRP_cap(facility_nodes=facility_nodes_np,
                                  demand_nodes=demand_nodes_np,
                                  solver=GUROBI_CMD(),
                                  fa_cap=fa_cap,
                                  de_demand=de_demand).prob_solve()
