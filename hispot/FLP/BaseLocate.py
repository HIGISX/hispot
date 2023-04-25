from pulp import *
import numpy as np


class Locate:
    def __init__(self, num_points, points, solver):
        self.num_points = num_points
        self.points = points
        self.solver = solver


class PModel(Locate):
    def __init__(self, num_points, points, solver, num_located):
        super().__init__(num_points, points, solver)
        self.name = None
        self.num_located = num_located
        self.centers = []
        self.assigns = {}

    def show_result(self, prob):
        prob.solve()
        print("Status:", LpStatus[prob.status])
        if LpStatus[prob.status] == "Optimal":
            obj = value(prob.objective)
            if self.name == "p-hub":
                for k in range(self.num_points):
                    if self.x[k][k].varValue == 1:
                        self.centers.append(k)
                        self.assigns[k] = []
                        for j in range(self.num_points):
                            if self.x[j][k].varValue == 1:
                                self.assigns[k].append(j)
                print("Selected Hubs =", self.centers)
                print("Assigned relationships = ", self.assigns)
                print("Minimum total cost =", obj)
                return self.centers, self.assigns, obj
            elif self.name == "p-dispersion":
                for i in range(self.num_points):
                    if self.x[i].varValue == 1:
                        self.centers.append(i)
                print("Centers =", self.centers)
                print("Maximum minimum distance between two points = ", value(prob.objective))
                return self.centers, obj
            else:
                for i in range(self.num_points):
                    if self.x[i].varValue == 1:
                        self.centers.append(i)
                        self.assigns[i] = []
                        for j in range(self.num_points):
                            if j != i and self.y[j][i].varValue == 1:
                                self.assigns[i].append(j)
                print("Centers =", self.centers)
                if self.name == 'p-median':
                    print("Assigned relationships = ", self.assigns)
                    print("Minimum total distance = ", value(prob.objective))
                elif self.name == 'p-center':
                    print("Assigned relationships = ", self.assigns)
                    print("Minimum Maximum distance = ", value(prob.objective))
                elif self.name == 'UFLP' or self.name == 'CFLP':
                    print("Assigned relationships = ", self.assigns)
                    print("Minimum total cost = ", value(prob.objective))
                # print(self.connections)
                return self.centers, self.assigns, obj
