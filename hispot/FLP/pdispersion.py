from hispot.FLP.BaseLocate import *


class PDispersion(PModel):
    def __init__(self, num_points, points, solver, num_located):
        super().__init__(num_points, points, solver, num_located)
        self.distance = None
        self.D_min = None
        self.x = None
        self.name = 'p-dispersion'

    def prob_solve(self):
        self.distance = np.sum((self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]) ** 2,
                               axis=-1) ** 0.5
        # use distance[i,j] to get the num
        M = 100
        # Create a new model
        prob = LpProblem("p-Dispersion_Problem", LpMaximize)
        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")  # X
        self.x = x
        D_min = LpVariable("min_Distance", lowBound=0, cat="Continuous")  # D_min

        # Set objective
        prob += D_min

        # Add constraints
        prob += (lpSum([x[i] for i in range(self.num_points)])) == self.num_located  # Fixed total number of facilities
        for i in range(self.num_points):
            for j in range(self.num_points):
                if i != j:
                    prob += (2 - x[i] - x[j]) * M + self.distance[i, j] >= D_min

        solve = self.show_result(prob)
        return solve
