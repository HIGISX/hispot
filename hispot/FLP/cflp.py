from hispot.FLP.BaseLocate import *


class CFLP(PModel):
    def __init__(self, cost, num_points, points, solver, num_located, demand, capacity):
        super().__init__(num_points, points, solver, num_located)
        self.distance = None
        self.cost = cost
        self.demand = demand
        self.capacity = capacity
        self.x = None
        self.y = None
        self.name = 'CFLP'

    def prob_solve(self):
        # distance = {(i, j): compute_distance(self.cover[i], self.cover[j]) for i, j in self.cartesian_prod}
        self.distance = np.sum((self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]) ** 2,
                               axis=-1) ** 0.5
        # Create a new model
        prob = LpProblem("CFLP", LpMinimize)

        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")  # X
        y = LpVariable.dicts("Assign", (Zones, Zones), lowBound=0, upBound=1, cat="Continuous")  # Y: Distribution received from each facility
        self.x = x
        self.y = y

        # Set objective
        prob += lpSum([[(y[i][j] * self.distance[i, j]) for i in range(self.num_points)]
                       for j in range(self.num_points)]) + \
                lpSum([(x[i] * self.cost[i] for i in range(self.num_points))])    # Minimum cost including shippingcost and establishment cost

        # Add constraints
        prob += (lpSum([x[i] for i in range(self.num_points)])) == self.num_located  # Fixed total number of facilities
        for i in range(self.num_points):
            prob += (lpSum([y[i][j] for j in range(self.num_points)])) == 1     #The total amount received from each facility is 1

        for j in range(self.num_points):
            prob += (lpSum([y[i][j] * self.demand[i] for i in range(self.num_points)])) <= self.capacity[j] * x[j]  # Demands need to be under the sum of capacity

        for j in range(self.num_points):
            for i in range(self.num_points):
                prob += y[i][j] <= x[j]  # Assign before locate; Points can only be assigned to facilities

        solve = self.show_result(prob)
        return solve
