from pulp import *


class FIFLP_Model:
    def __init__(self, num_path, num_vector, path_vector, solver):
        self.name = None
        self.num_path = num_path
        self.num_vector = num_vector
        self.path_vector = path_vector
        self.solver = solver

    def show_result(self, prob):
        prob.solve(self.solver)
        print("Status:", LpStatus[prob.status])
        selected_path = []
        selected_vector = []
        if LpStatus[prob.status] == "Optimal":
            if self.name == 'MaFM':
                for i in range(self.num_path):
                    if self.xp[i].varValue == 1:
                        selected_path.append(i)
                for i in range(self.num_vector):
                    if self.yi[i].varValue == 1:
                        selected_vector.append(i)
                print("Selected paths =", selected_path)
                print("Selected points =", selected_vector)
                print("Maximum flow =", value(prob.objective))
            elif self.name == 'MiFM':
                for i in range(self.num_path):
                    if self.xp[i].varValue == 1:
                        selected_path.append(i)
                for i in range(self.num_vector):
                    if self.yi[i].varValue == 1:
                        selected_vector.append(i)
                print("Selected paths =", selected_path)
                print("Selected points =", selected_vector)
                print("Minimum flow =", value(prob.objective))
            elif self.name == 'MaAG':
                for i in range(self.num_path):
                    for j in self.path_vector[i]:
                        if self.xpi[i, j].varValue == 1:
                            selected_path.append(i)
                            break
                for i in range(self.num_vector):
                    if self.yi[i].varValue == 1:
                        selected_vector.append(i)
                print("Selected paths =", selected_path)
                print("Selected points =", selected_vector)
                print("Maximum flow =", value(prob.objective))
            elif self.name == 'MiFMG':
                for i in range(self.num_path):
                    for j in self.path_vector[i]:
                        if self.xpi[i, j].varValue == 1:
                            selected_path.append(i)
                            break
                for i in range(self.num_vector):
                    if self.yi[i].varValue == 1:
                        selected_vector.append(i)
                print("Selected paths =", selected_path)
                print("Selected points =", selected_vector)
                print("Minimization of the Number of Facilities for Gain Maximization =", value(prob.objective))
        return selected_path, selected_vector