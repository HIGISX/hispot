from hispot.LRP.BaseLRP import *
import numpy as np


class LRP_cap(LRP_Model):

    def __init__(self, facility_nodes, demand_nodes, solver,  fa_cap, de_demand):
        super().__init__(facility_nodes, demand_nodes, solver)
        self.distance = None
        self.fa_cap = fa_cap
        self.de_demand = de_demand
        self.x = None
        self.y = None
        self.name = 'LRP_cap'

    def prob_solve(self):
        self.distance = np.sum((self.facility_nodes[:, np.newaxis, :] - self.demand_nodes[np.newaxis, :, :]) ** 2, axis=-1) ** 0.5
        # Create a new model
        prob = LpProblem("Location_Routing_Problem", LpMinimize)
        num_fa = len(self.facility_nodes)
        num_de = len(self.demand_nodes)

        # Create variables
        set_F = list(range(num_fa))
        set_D = list(range(num_de))
        y = LpVariable.dicts("Select", set_F, cat="Binary")  # Y
        x = LpVariable.dicts("Assign", (set_F, set_D), cat="Binary")  # X
        self.y = y
        self.x = x

        # Add constraints
        for i in range(num_fa):
            prob += (lpSum([x[i][j] * self.de_demand[j] for j in range(num_de)]) <= self.fa_cap[i][1] * y[i])

        for i in range(num_fa):
            prob += (lpSum([x[i][j] * self.de_demand[j] for j in range(num_de)]) >= self.fa_cap[i][0] * y[i])

        constraints_eq = {j: prob.addConstraint(
            LpConstraint(
                e=lpSum(x[i][j] for i in range(num_fa)),
                sense=LpConstraintEQ,
                rhs=1,
                name="constraint_eq_{0}".format(j)
            )
        ) for j in range(num_de)}

        # Set objective
        # prob += lpSum([[x[i][j] * self.distance[i, j] for i in set_F] for j in set_D])  # Minimum total distance
        prob += lpSum([[x[i][j] * self.distance[i, j] for i in range(num_fa)] for j in range(num_de)])  # Minimum total distance
        solver = self.show_result(prob)
        return solver
