from hispot.coverage.BaseCoverage import *
import numpy as np

class MCLP(CModel):
    def __init__(self, num_located, num_people, num_points, points, radius, solver):
        super().__init__(num_points, points, solver)
        self.distance = None
        self.num_located = num_located
        self.num_people = num_people
        self.radius = radius
        self.x = None
        self.z = None
        self.name = 'MCLP'

    def prob_solve(self):
        self.distance = np.sum((self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]) ** 2, axis=-1) ** 0.5
        mask = self.distance <= self.radius
        self.distance[mask] = 1
        self.distance[~mask] = 0

        # Create a new model
        prob = LpProblem("Maximum_Covering_Model", LpMaximize)

        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")  # X
        z = LpVariable.dicts("Serve", Zones, cat="Binary")  # Z
        self.x = x
        self.z = z
        # Set objective
        prob += lpSum(z[i] * self.num_people[i] for i in range(self.num_points))  # Maximum number of people served

        # Add constraints
        prob += (lpSum([x[j] for j in range(self.num_points)]) == self.num_located)
        for i in range(self.num_points):
            prob += (lpSum([x[j] * self.distance[i, j] for j in range(self.num_points)]) >= z[i])

        solve = self.show_result(prob)
        return solve
