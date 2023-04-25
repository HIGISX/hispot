from hispot.FLP.BaseLocate import *
import numpy as np


class PHub(PModel):
    def __init__(self, num_points, points, solver, num_located, weight, collect_cost, transfer_cost, distribution_cost):
        super().__init__(num_points, points, solver, num_located)
        self.distance = None
        self.num_hubs = num_located
        self.PC = collect_cost
        self.PT = transfer_cost
        self.PD = distribution_cost
        self.weight = weight
        self.Oi = np.sum(self.weight, 1)  # 节点的起始货流量
        self.Di = np.sum(self.weight, 0)  # 节点的终止货流量
        self.x = None
        self.y = None
        self.name = 'p-hub'

    def prob_solve(self):
        ## Compute distance matrix
        self.distance = np.sum((self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]) ** 2, axis=-1) ** 0.5

        # Add variable
        prob = LpProblem("P_HUB", LpMinimize)
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("x", (Zones, Zones), cat="Binary")  # Y
        y = LpVariable.dicts("y", (Zones, Zones, Zones), cat="Continuous", lowBound=0)  # Y
        N = range(self.num_points)
        self.x = x
        self.y = y
        # Set objective
        prob += lpSum(self.PC * self.distance[i][k] * x[i][k] * self.Oi[i] for i in N for k in N) + lpSum(
            self.PT * self.distance[k][h] * y[i][k][h] for i in N for k in N for h in N) + lpSum(
            self.PD * self.distance[k][i] * x[i][k] * self.Di[i] for i in N for k in N)
        # Add constraints
        for i in N:
            prob += lpSum(x[i][k] for k in N) == 1
            for k in N:
                prob += x[i][k] <= x[k][k]
                prob += (lpSum(y[i][k][h] for h in N) - lpSum(y[i][h][k] for h in N)) == (
                        self.Oi[i] * x[i][k] - lpSum(self.weight[i][j] * x[j][k] for j in N))
                prob += lpSum(y[i][k][h] for h in N) <= self.Oi[i] * x[i][k]

        prob += lpSum(x[k][k] for k in N) == self.num_hubs

        solver = self.show_result(prob)
        return solver
