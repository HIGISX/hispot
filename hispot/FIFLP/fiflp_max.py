from hispot.FIFLP.BaseFIFLP import *
import numpy as np


class MaFM(FIFLP_Model):
    def __init__(self, num_path, num_vector, path_vector, num_choice, path_flow, solver):
        super().__init__(num_path, num_vector, path_vector, solver)
        self.num_choice = num_choice
        self.path_flow = path_flow
        self.name = 'MaFM'
        self.xp = None
        self.yi = None

    def prob_solve(self):
        # Build a problem model
        prob = LpProblem("Maximization of the Intercepted Flow", LpMaximize)

        # Create variables
        zones_p = list(range(self.num_path))
        zones_v = list(range(self.num_vector))
        xp = LpVariable.dicts("Select_Path", zones_p, cat="Binary")  # xp
        yi = LpVariable.dicts("Select_Vector", zones_v, cat="Binary")  # yi
        self.xp = xp
        self.yi = yi

        # Set objective
        prob += pulp.lpSum(self.path_flow[i] * self.xp[i] for i in range(self.num_path))

        # Add constraints
        prob += pulp.lpSum(yi[i] for i in range(self.num_vector)) == self.num_choice

        for k in range(self.num_path):
            prob += pulp.lpSum(yi[i] for i in self.path_vector[k]) >= self.xp[k]

        # solve
        solve = self.show_result(prob)
        return solve
