# FILE: nsga2_facility_location.py
# This file defines the NSGA-II algorithm as a reusable class.

import numpy as np
import random
import time
from matplotlib import pyplot as plt


class NSGAIIForFacilityLocation:
    """
    Implementation of the NSGA-II algorithm for the facility location problem.
    This algorithm is designed to optimize two conflicting objectives:
    1. Maximize the total service benefit (based on demand and distance function F).
    2. Minimize the total travel distance for all users.
    """

    def __init__(self, n, m, p, users, facilities, demand, w1, w2, da, db, D_max, A, dist,
                 population_size=100, generations=200, mutation_prob=0.2):
        """Initializes the NSGA-II algorithm parameters."""
        self.n = n
        self.m = m
        self.p = p
        self.users = users
        self.facilities = facilities
        self.demand = demand
        self.w1 = w1
        self.w2 = w2
        self.da = da
        self.db = db
        self.D_max = D_max
        self.A = A
        self.dist_matrix = dist
        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.pareto_front = None
        self.execution_time = None

    def F_vectorized(self, d):
        """Vectorized version of the zero-tolerance distance function F(dij)."""
        res = np.zeros_like(d, dtype=float)
        mask1 = d <= self.da
        res[mask1] = 1.0
        mask2 = (d > self.da) & (d <= self.db)
        if self.db > self.da:
            res[mask2] = 0.5 * np.cos(
                (np.pi / (self.db - self.da)) * (d[mask2] - (self.da + self.db) / 2) + np.pi / 2) + 0.5
        return res

    def calculate_objectives(self, chromosome):
        """
        Calculates the two normalized objectives and coverage rate for a given chromosome.
        Obj1 (minimize): -w1 * sum(demand*F(d)) / total_demand  (effective coverage benefit)
        Obj2 (minimize):  w2 * sum(demand*d) / (db*total_demand)  (distance burden)
        Returns: tuple: (obj1, obj2, coverage_rate).
        """
        chromosome = list(chromosome)
        selected_dists = self.dist_matrix[chromosome, :]
        selected_A = self.A[chromosome, :]
        F_matrix = self.F_vectorized(selected_dists)

        total_demand = np.sum(self.demand)

        # Old objectives:
        # potential_benefits = (self.w1 * self.demand * F_matrix)
        # potential_costs = (self.w2 * selected_dists)

        # New normalized objectives
        potential_benefits = self.w1 * self.demand * F_matrix / total_demand
        potential_costs = self.w2 * self.demand * selected_dists / (self.db * total_demand)

        net_scores = potential_benefits - potential_costs
        net_scores[selected_A == 0] = -np.inf

        best_facility_indices = np.argmax(net_scores, axis=0)
        covered_users_mask = np.any(selected_A > 0, axis=0)

        total_benefit, total_cost, covered_demand = 0, 0, 0

        if np.any(covered_users_mask):
            best_benefits = potential_benefits[best_facility_indices[covered_users_mask], covered_users_mask]
            best_costs = potential_costs[best_facility_indices[covered_users_mask], covered_users_mask]
            total_benefit = np.sum(best_benefits)
            total_cost = np.sum(best_costs)
            covered_demand = np.sum(self.demand[covered_users_mask])

        coverage_rate = covered_demand / total_demand if total_demand > 0 else 0
        return -total_benefit, total_cost, coverage_rate

    @staticmethod
    def fast_non_dominated_sort(objectives):
        """Performs fast non-dominated sorting."""
        n_solutions = len(objectives)
        S = [[] for _ in range(n_solutions)]
        fronts = [[]]
        n = [0] * n_solutions

        for p in range(n_solutions):
            for q in range(n_solutions):
                if p == q: continue
                if (objectives[p][0] <= objectives[q][0] and objectives[p][1] < objectives[q][1]) or \
                        (objectives[p][0] < objectives[q][0] and objectives[p][1] <= objectives[q][1]):
                    if q not in S[p]: S[p].append(q)
                elif (objectives[q][0] <= objectives[p][0] and objectives[q][1] < objectives[p][1]) or \
                        (objectives[q][0] < objectives[p][0] and objectives[q][1] <= objectives[p][1]):
                    n[p] += 1
            if n[p] == 0: fronts[0].append(p)

        i = 0
        while fronts[i]:
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0: Q.append(q)
            i += 1
            fronts.append(Q)
        del fronts[-1]
        return fronts

    @staticmethod
    def calculate_crowding_distance(front_indices, objectives):
        """Calculates crowding distance for a given front."""
        num_points = len(front_indices)
        if num_points <= 2: return {idx: float('inf') for idx in front_indices}

        distances = {idx: 0.0 for idx in front_indices}
        front_objectives = [objectives[i] for i in front_indices]

        for m in range(len(objectives[0])):
            sorted_front = sorted(zip(front_indices, front_objectives), key=lambda x: x[1][m])
            obj_min, obj_max = sorted_front[0][1][m], sorted_front[-1][1][m]
            if obj_max == obj_min: continue

            distances[sorted_front[0][0]] = float('inf')
            distances[sorted_front[-1][0]] = float('inf')

            for i in range(1, num_points - 1):
                distances[sorted_front[i][0]] += (sorted_front[i + 1][1][m] - sorted_front[i - 1][1][m]) / (
                            obj_max - obj_min)
        return distances

    def crossover(self, parent1, parent2):
        """Performs set-based crossover."""
        p1_set, p2_set = set(parent1), set(parent2)
        common = list(p1_set & p2_set)
        unique_genes = list((p1_set - p2_set) | (p2_set - p1_set))
        random.shuffle(unique_genes)

        child1, child2 = common[:], common[:]

        needed = self.p - len(common)
        child1.extend(unique_genes[:needed])
        child2.extend(unique_genes[needed:])  # Take remaining unique

        # Ensure correct length and diversity
        child1_set = set(child1)
        while len(child1) < self.p:
            gene = random.randrange(self.m)
            if gene not in child1_set: child1.append(gene); child1_set.add(gene)

        child2_set = set(child2)
        while len(child2) < self.p:
            gene = random.randrange(self.m)
            if gene not in child2_set: child2.append(gene); child2_set.add(gene)

        return child1[:self.p], child2[:self.p]

    def mutate(self, chromosome):
        """Performs mutation on a chromosome."""
        if random.random() < self.mutation_prob:
            idx_to_replace = random.randrange(self.p)
            options = [i for i in range(self.m) if i not in set(chromosome)]
            if options: chromosome[idx_to_replace] = random.choice(options)
        return chromosome

    def optimize(self):
        """Main optimization loop for the NSGA-II algorithm."""
        start_time = time.time()
        population = [random.sample(range(self.m), self.p) for _ in range(self.population_size)]

        for gen in range(self.generations):
            offspring_population = []
            while len(offspring_population) < self.population_size:
                p1, p2 = random.sample(population, 2)
                c1, c2 = self.crossover(p1, p2)
                offspring_population.append(self.mutate(c1))
                if len(offspring_population) < self.population_size:
                    offspring_population.append(self.mutate(c2))

            combined_population = population + offspring_population
            objectives_with_coverage = [self.calculate_objectives(ind) for ind in combined_population]
            objectives_for_sorting = [t[:2] for t in objectives_with_coverage]
            fronts = self.fast_non_dominated_sort(objectives_for_sorting)

            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend([combined_population[i] for i in front])
                else:
                    distances = self.calculate_crowding_distance(front, objectives_for_sorting)
                    sorted_front = sorted(front, key=lambda idx: distances[idx], reverse=True)
                    remaining = self.population_size - len(new_population)
                    new_population.extend([combined_population[i] for i in sorted_front[:remaining]])
                    break
            population = new_population

            if (gen + 1) % 50 == 0: print(f"Generation {gen + 1}/{self.generations}...")

        final_results = [self.calculate_objectives(ind) for ind in population]
        final_objectives_for_sorting = [t[:2] for t in final_results]
        final_fronts = self.fast_non_dominated_sort(final_objectives_for_sorting)

        self.pareto_front = []
        if final_fronts:
            for idx in final_fronts[0]:
                res = final_results[idx]
                self.pareto_front.append({
                    "chromosome": population[idx],
                    "objectives": (-res[0], res[1]),
                    "coverage": res[2]
                })

        self.execution_time = time.time() - start_time
        print("\nOptimization finished.")
        return self.pareto_front, self.execution_time

    def plot_pareto_front(self):
        """Plots the resulting Pareto front."""
        if not self.pareto_front:
            print("No Pareto front to plot. Run optimize() first.")
            return

        benefits = [sol['objectives'][0] for sol in self.pareto_front]
        costs = [sol['objectives'][1] for sol in self.pareto_front]

        plt.figure(figsize=(10, 6))
        plt.scatter(costs, benefits, c='blue', marker='o')
        plt.title('Pareto Front for Facility Location')
        plt.xlabel('Objective 2: Total Travel Cost (Minimized)')
        plt.ylabel('Objective 1: Total Service Benefit (Maximized)')
        plt.grid(True)
        plt.show()