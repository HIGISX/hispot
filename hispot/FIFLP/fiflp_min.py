from hispot.FIFLP.BaseFIFLP import *
import numpy as np


class MiFM(FIFLP_Model):
    def __init__(self, num_path, num_vector, path_vector, path_flow, intercept_e, solver):
        super().__init__(num_path, num_vector, path_vector, solver)
        self.intercept_e = intercept_e
        self.path_flow = path_flow
        self.name = 'MiFM'
        self.xp = None
        self.yi = None

    def prob_solve(self):
        # Build a problem model
        prob = LpProblem("Minimization of the Intercepted Flow", LpMinimize)

        # Create variables
        zones_p = list(range(self.num_path))
        zones_v = list(range(self.num_vector))
        xp = LpVariable.dicts("Select_Path", zones_p, cat="Binary")  # xp
        yi = LpVariable.dicts("Select_Vector", zones_v, cat="Binary")  # yi
        self.xp = xp
        self.yi = yi

        # Set objective
        prob += pulp.lpSum(yi[i] for i in range(self.num_vector))

        # Add constraints
        for p in range(self.num_path):
            prob += pulp.lpSum(yi[i] for i in self.path_vector[p]) >= xp[p]

        prob += pulp.lpSum(self.path_flow[p] * xp[p] for p in range(self.num_path)) >= self.intercept_e

        solve = self.show_result(prob)
        return solve
