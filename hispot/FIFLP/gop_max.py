from hispot.FIFLP.BaseFIFLP import *
import numpy as np


class MaAG(FIFLP_Model):
    def __init__(self, num_path, num_vector, path_vector, num_choice, vector_gain, solver):
        super().__init__(num_path, num_vector, path_vector, solver)
        self.num_choice = num_choice
        self.vector_gain = vector_gain
        self.name = 'MaAG'
        self.xpi = None
        self.yi = None

    def prob_solve(self):
        # Build a problem model
        prob = LpProblem("Maximization of the Achievable Gain", LpMaximize)

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
        prob += pulp.lpSum(self.vector_gain[p, i % 2] * xpi[p, i]
                           for p in range(self.num_path) for i in self.path_vector[p])

        # Add constraints
        prob += pulp.lpSum(yi[i] for i in range(self.num_vector)) == self.num_choice

        for p in range(self.num_path):
            for k in self.path_vector[p]:
                prob += pulp.lpSum(yi[i] for i in self.path_vector[p]) >= xpi[p, k]

        for p in range(self.num_path):
            prob += pulp.lpSum(xpi[p, i] for i in self.path_vector[p]) <= 1

        # solve
        solve = self.show_result(prob)
        return solve
