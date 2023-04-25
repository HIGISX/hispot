import numpy as np
import random
from hispot.LRP import LRP_cost
from pulp import *

random.seed(1)
num_fa = 5
num_de = 20
facility_nodes = [(random.random(), random.random()) for i in range(num_fa)]
fa_cap = [(random.randint(30, 35), random.randint(35, 40)) for i in range(num_fa)]
demand_nodes = [(random.random(), random.random()) for i in range(num_de)]
de_demand = [random.randint(1, 10) for i in range(num_de)]
fa_cost = [random.randint(0, 1) for i in range(num_fa)]

facility_nodes_np = np.array(facility_nodes)
demand_nodes_np = np.array(demand_nodes)

selected, assigned, obj = LRP_cost(facility_nodes=facility_nodes_np,
                                   demand_nodes=demand_nodes_np,
                                   solver=GUROBI_CMD(),
                                   fa_cap=fa_cap,
                                   de_demand=de_demand,
                                   fa_cost=fa_cost).prob_solve()
