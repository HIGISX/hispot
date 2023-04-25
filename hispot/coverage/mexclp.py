from hispot.coverage.BaseCoverage import *
import numpy as np


class MEXCLP(CModel):
    def __init__(self, num_located, demand, num_points, points, radius, unprob_rate, solver):
        super().__init__(num_points, points, solver)
        self.distance = None
        self.num_located = num_located
        self.demand = demand
        self.radius = radius
        self.unprob_rate = unprob_rate
        self.x = None
        self.y = None
        self.name = 'MEXCLP'

    def prob_solve(self):
        self.distance = np.sum((self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]) ** 2, axis=-1) ** 0.5
        mask = self.distance <= self.radius
        self.distance[mask] = 1
        self.distance[~mask] = 0

        # Create a new model
        prob = pulp.LpProblem("MEXCLP", LpMaximize)
        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")  # X
        y = LpVariable.dicts("Assign", (Zones, Zones), cat="Binary")  # Y
        self.y = y
        self.x = x

        # Set objective
        prob += lpSum([[(1 - self.unprob_rate) * (self.unprob_rate ** k) * self.demand[i] * y[i][k]
                       for i in range(self.num_points)] for k in range(self.num_located)])

        # Add constraints
        for i in range(self.num_points):
            prob+=(lpSum([x[j] * self.distance[i, j] for j in range(self.num_points)]) - lpSum([k * y[i][k] for k in range(self.num_located)]))

        prob += (lpSum([x[j] for j in range(self.num_points)]) <= self.num_located)

        solve = self.show_result(prob)
        return solve
