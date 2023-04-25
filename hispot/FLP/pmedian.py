import numpy as np
import random
from hispot.FLP.BaseLocate import *


class PMedian(PModel):
    def __init__(self, num_points, points, solver, num_located):
        super().__init__(num_points, points, solver, num_located)
        self.distance = None
        self.x = None
        self.y = None
        self.name = 'p-median'

    def prob_solve(self):
        self.distance = np.sum((self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]) ** 2,
                               axis=-1) ** 0.5
        # use distance[i,j] to get the num
        # Create a new model
        prob = LpProblem("p-Median_Problem", LpMinimize)

        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")  # X
        y = LpVariable.dicts("Assign", (Zones, Zones), cat="Binary")  # Y
        self.y = y
        self.x = x
        # Set objective
        prob += lpSum([[y[i][j] * self.distance[i, j] for i in range(self.num_points)] for j in
                       range(self.num_points)])  # Minimum total distance

        # Add constraints
        prob += (lpSum([x[i] for i in range(self.num_points)])) == self.num_located  # Fixed total number of facilities

        for i in range(self.num_points):
            prob += (lpSum(
                [y[i][j] for j in range(self.num_points)])) == 1  # Each point only corresponds to one facility

        for j in range(self.num_points):
            for i in range(self.num_points):
                prob += y[i][j] <= x[j]  # Assign before locate; Points can only be assigned to facilities

        solve = self.show_result(prob)
        return solve

    def teitz_bart(self):
        self.distance = np.sum((self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]) ** 2,
                               axis=-1) ** 0.5
        N = self.num_points
        p = self.num_located
        median = random.sample(range(N), p)
        d1 = [-1 for i in range(N)]
        d2 = [-1 for i in range(N)]

        # update_assignment
        node1, node2 = -1, -1
        for i in range(N):
            dist1, dist2 = np.inf, np.inf
            for j in range(p):
                if self.distance[i][median[j]] < dist1:
                    dist2 = dist1
                    node2 = node1
                    dist1 = self.distance[i][median[j]]
                    node1 = median[j]
                elif self.distance[i][median[j]] < dist2:
                    dist2 = self.distance[i][median[j]]
                    node2 = median[j]
            d1[i] = node1
            d2[i] = node2
        dist1 = 0
        for i in range(N):
            dist1 += self.distance[i][d1[i]]
        r = dist1

        verbose = True
        if verbose:
            print(r)
        while True:
            result = next(self.distance, median, d1, d2, p, N)
            if result[0]:
                r = result[1]
                if verbose:
                    print(r)
            else:
                break
        return median, r

    # def heuristic_solver(self):
    #     x = [0] * self.num_points
    #     y = [[0 for j in range(self.num_points)] for i in range(self.num_points)]
    #     self.distance = {(i, j): compute_distance(self.cover[i], self.cover[j]) for i, j in self.cartesian_prod}
    #     self.unselected_points = list(range(self.num_points))
    #     # Greedy algorithm for selecting facilities
    #     for _ in range(self.num_located):
    #         max_demand = -1
    #         max_demand_index = -1
    #         for i in self.unselected_points:
    #             demand = sum([self.distance[i, j] * self.num_people[j] for j in range(self.num_points)])
    #             if demand > max_demand:
    #                 max_demand = demand
    #                 max_demand_index = i
    #         self.selected.append(max_demand_index)
    #         self.unselected_points.remove(max_demand_index)
    #         x[max_demand_index] = 1
    #
    #     # Assign each point to the nearest selected facility
    #     for i in range(self.num_points):
    #         min_distance = float('inf')
    #         min_distance_index = -1
    #         for j in self.selected:
    #             if self.distance[i, j] < min_distance:
    #                 min_distance = self.distance[i, j]
    #                 min_distance_index = j
    #         y[i][min_distance_index] = 1
    #
    #     self.selected_points = [i for i in range(self.num_points) if x[i] == 1]
    #     self.unselected_points = [i for i in range(self.num_points) if x[i] == 0]
    #     self.connections = [(i, j) for i in range(self.num_points) for j in range(self.num_points) if y[i][j] == 1]
    #
    #     return self.selected, self.selected_points, self.unselected_points, self.connections