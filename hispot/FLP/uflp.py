from hispot.FLP.BaseLocate import *


class UFLP(PModel):
    def __init__(self, cost, num_points, points, solver, num_located):
        super().__init__(num_points, points, solver, num_located)
        self.cost = cost
        self.distance = None
        self.x = None
        self.y = None
        self.name = 'UFLP'

    def prob_solve(self):
        self.distance = np.sum((self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]) ** 2,
                               axis=-1) ** 0.5

        # Create a new model
        prob = LpProblem("UFLP", LpMinimize)

        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")  # X
        y = LpVariable.dicts("Assign", (Zones, Zones), cat="Binary")  # Y
        self.x = x
        self.y = y

        # Set objective
        prob += lpSum([[(y[i][j] * self.distance[i, j]) for i in range(self.num_points)]
                       for j in range(self.num_points)]) + \
                lpSum([(x[i] * self.cost[i] for i in range(self.num_points))])    # Minimum cost including shippingcost and establishment cost

        # Add constraints
        prob += (lpSum([x[i] for i in range(self.num_points)])) <= self.num_located     # Under than total number of facilities
        for i in range(self.num_points):
            prob += (lpSum([y[i][j] for j in range(self.num_points)])) == 1     # Each point only corresponds to one facility

        for j in range(self.num_points):
            for i in range(self.num_points):
                prob += y[i][j] <= x[j]  # Assign before locate; Points can only be assigned to facilities

        solve = self.show_result(prob)
        return solve
