from pulp import *


class CModel:
    def __init__(self, num_points, points, solver):
        self.num_points = num_points
        self.points = points
        self.solver = solver
        self.name = None
        self.centers = []

    def show_result(self, prob):
        prob.solve()
        print("Status:", LpStatus[prob.status])
        if LpStatus[prob.status] == "Optimal":
            for i in range(self.num_points):
                if self.x[i].varValue == 1:
                    self.centers.append(i)
        obj = value(prob.objective)
        if self.name == 'MCLP' or self.name == "LSCP":
            print("Selected points =", self.centers)
            print("The coverage radius =", self.radius)
            print("Minimum cost =", obj)
            return self.centers, obj
        elif self.name == "BCLP":
            cover_twice = []
            for i in range(self.num_points):
                if self.u[i].varValue == 1:
                    cover_twice.append(i)
            print("Selected points =", self.centers)
            print("Covered twice points =", cover_twice)
            print("The objective is = ", obj)
            return self.centers, cover_twice, obj
        elif self.name == 'MEXCLP':
            print("Selected points =", self.centers)
            print("The objective is = ", obj)
            return self.centers, obj
