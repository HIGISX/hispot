from hispot.coverage.BaseCoverage import *
import numpy as np


class BCLP(CModel):
    def __init__(self, num_located, num_points, points, radius, setup_cost, solver):
        super().__init__(num_points, points, solver)
        self.setup_cost = setup_cost
        self.distance = None
        self.num_located = num_located
        self.radius = radius
        self.x = None
        self.y = None
        self.u = None

        self.name = 'BCLP'

    def prob_solve(self):
        self.distance = np.sum((self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]) ** 2, axis=-1) ** 0.5
        mask = self.distance <= self.radius
        self.distance[mask] = 1
        self.distance[~mask] = 0

        # Build a problem model
        prob = LpProblem("Backup_Coverage_Location_Problem", LpMaximize)

        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")
        y = LpVariable.dicts("Serve", Zones, cat="Binary")
        u = LpVariable.dicts("Twice", Zones, cat="Binary")
        self.x = x
        self.y = y
        self.u = u

        Z1 = lpSum(y[i] * self.setup_cost[i] for i in range(self.num_points))
        Z2 = lpSum(u[i] * self.setup_cost[i] for i in range(self.num_points))
        # Set objective
        prob += Z1 + Z2
        for i in range(self.num_points):
            prob += lpSum([x[j] * self.distance[i, j] for j in range(self.num_points)]) - y[i] - u[i] >= 0
            prob += u[i] - y[i] <= 0

        prob += lpSum(x[i] for i in range(self.num_points)) == self.num_located

        solve = self.show_result(prob)
        return solve


