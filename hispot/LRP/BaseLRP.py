from pulp import *


# class LocateLRP:
#     def __init__(self, facility_nodes, demand_nodes, solver):
#         self.facility_nodes = facility_nodes
#         self.demand_nodes = demand_nodes
#         self.solver = solver


class LRP_Model:
    def __init__(self, facility_nodes, demand_nodes, solver):
        self.name = None
        self.facility_nodes = facility_nodes
        self.demand_nodes = demand_nodes
        self.solver = solver

    def show_result(self, prob):
        global selected, selected_facility, unselected, unselected_facility, assigned
        # selected, selected_facility, unselected, unselected_facility, assigned = [], [], [], [], []
        selected, unselected = [], []
        assigned = {}
        prob.solve(self.solver)
        print("Status:", LpStatus[prob.status])
        if LpStatus[prob.status] == "Optimal":
            for i in range(len(self.facility_nodes)):
                if self.y[i].varValue == 1:
                    selected.append(i)
                    assigned[str(i)] = []
                    # selected_facility.append(self.facility_nodes[i])
                    for j in range(len(self.demand_nodes)):
                        if self.x[i][j].varValue == 1:
                            assigned[str(i)].append(j)
                else:
                    unselected.append(i)
                    # unselected_facility.append(self.facility_nodes[i])

        print("Selected facilities =", selected)
        print("Unselected facilities =", unselected)
        print("Assigned relationships = ", assigned)
        print("Minimum total distance = ", value(prob.objective))

        return selected, assigned, value(prob.objective)