from hispot.FIFLP.BaseFIFLP import *
import numpy as np


class MiFMG(FIFLP_Model):
    def __init__(self, num_path, num_vector, path_vector, vector_gain, flow_gain, solver):
        super().__init__(num_path, num_vector, path_vector, solver)
        self.vector_gain = vector_gain
        self.flow_gain = flow_gain
        self.name = 'MiFMG'
        self.xpi = None
        self.yi = None

    def prob_solve(self):
        # Build a problem model
        prob = LpProblem('Minimization of the Number of Facilities for Gain Maximization', LpMinimize)

        # Create variables
        xpi = {}
        for i in range(self.num_path):
            for j in self.path_vector[i]:
                name = 'Select_Path' + str(i) + '_' + str(j)
                xpi[i, j] = pulp.LpVariable(name, 0, 1, LpBinary)
        zones_v = list(range(self.num_vector))
        yi = LpVariable.dicts("Select_Vector", zones_v, cat="Binary")  # yi
        self.xpi = xpi
        self.yi = yi

        # Set objective
        prob += pulp.lpSum(yi[i] for i in range(self.num_vector))

        # Add constraints
        for p in range(self.num_path):
            for k in self.path_vector[p]:
                prob += pulp.lpSum(yi[i] for i in self.path_vector[p]) >= xpi[p, k]

        for p in range(self.num_path):
            prob += pulp.lpSum(xpi[p, i] for i in self.path_vector[p]) <= 1

        prob += pulp.lpSum(self.vector_gain[p, i % 2] * xpi[p, i]
                           for p in range(self.num_path) for i in self.path_vector[p]) >= self.flow_gain

        # solve
        solve = self.show_result(prob)
        return solve
