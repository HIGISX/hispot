from hispot.coverage.BaseCoverage import *
import numpy as np

class LSCP(CModel):
    def __init__(self, num_points, points, solver, radius):
        super().__init__(num_points, points, solver)
        self.x = None
        self.radius = radius
        self.name = 'LSCP'

    def prob_solve(self):
        self.distance = np.sum((self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]) ** 2, axis=-1) ** 0.5
        mask = self.distance <= self.radius
        self.distance[mask] = 1
        self.distance[~mask] = 0

        # Create a new model
        prob = LpProblem("Location_Set_Covering_Model", LpMinimize)

        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")  # X
        self.x = x

        # Set objective
        prob += lpSum(x[i] for i in range(self.num_points))  # Minimum total cost
        # Add constraints
        for j in range(self.num_points):
            prob += (lpSum([self.distance[i, j] * x[i] for i in range(self.num_points)]) >= 1)

        solve = self.show_result(prob)
        return solve
