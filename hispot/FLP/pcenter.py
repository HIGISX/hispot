from hispot.FLP.BaseLocate import *


class PCenter(PModel):
    def __init__(self, num_points, points, solver, num_located):
        super().__init__(num_points, points, solver, num_located)
        self.distance = None
        self.x = None
        self.y = None
        self.name = 'p-center'

    def prob_solve(self):
        self.distance = np.sum((self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]) ** 2,
                               axis=-1) ** 0.5
        # use distance[i,j] to get the num
        # Create a new model
        prob = LpProblem("p-Center_Problem", LpMinimize)

        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")  # X
        y = LpVariable.dicts("Assign", (Zones, Zones), cat="Binary")  # Y
        self.y = y
        self.x = x
        Z = LpVariable("max_Distance", lowBound=0, cat="Continuous")
        # Set objective
        prob += Z

        # Add constraints
        prob += (lpSum([x[i] for i in range(self.num_points)])) == self.num_located  # Fixed total number of facilities

        for i in range(self.num_points):
            prob += (lpSum(
                [y[i][j] for j in range(self.num_points)])) == 1  # Each point only corresponds to one facility

        for j in range(self.num_points):
            for i in range(self.num_points):
                prob += y[i][j] <= x[j]  # Assign before locate; Points can only be assigned to facilities

        for i in range(self.num_points):
            prob += lpSum([y[i][j] * self.distance[i, j] for j in range(self.num_points)]) <= Z
        # prob += [lpSum(y[i][j] * self.distance[i, j] for i in range(self.num_points)) <= Z for j in range(self.num_points)]
        solve = self.show_result(prob)
        return solve
