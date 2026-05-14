# import random
# import copy
# import numpy as np
# import time
# from matplotlib import pyplot as plt
# import torch
# from matplotlib.patches import Circle
#
# class Chromosome:
#     def __init__(self, content, fitness):
#         self.content = content
#         self.fitness = fitness
#
#     def __str__(self): return "%s f=%.2f" % (self.content, self.fitness)
#     def __repr__(self): return "%s f=%.2f" % (self.content, self.fitness)
#
# class GeneticAlgorithmVectorized:
#     def __init__(self, n, m, p, users, facilities, demand, w1, w2, da, db, D_max, A, dist,
#                  population_size=100, generations=200, mutation_prob=0.2, elite_size=5):
#         self.n = n
#         self.m = m
#         self.p = p
#         self.users = users
#         self.facilities = facilities
#         self.demand = demand
#         self.w1 = w1
#         self.w2 = w2
#         self.da = da
#         self.db = db
#         self.D_max = D_max
#         self.A = A
#         self.dist_matrix = dist
#
#         self.population_size = population_size
#         self.generations = generations
#         self.mutation_prob = mutation_prob
#         self.elite_size = elite_size
#
#         self.best_solution = None
#         self.best_fitness = -np.inf
#         self.best_coverage = 0
#         self.best_penalty = 0
#
#     def F_vectorized(self, d):
#         res = np.zeros_like(d)
#         res[d <= self.da] = 1.0
#         mask = (d > self.da) & (d <= self.db)
#         # 确保分母不为零
#         if self.db > self.da:
#             res[mask] = 0.5 + 0.5 * np.cos(
#                  (np.pi / (self.db - self.da)) * (d[mask] - (self.da + self.db) / 2) + np.pi / 2)
#         return res
#
#     def fitness(self, chromosome):
#         """
#                Calculates the fitness of a chromosome based on the objective function.
#                A user is served by the single best facility that covers it.
#                """
#         selected_dists = self.dist_matrix[chromosome, :]
#         selected_A = self.A[chromosome, :]
#
#         F_matrix = self.F_vectorized(selected_dists)
#
#         # 计算每个设施对每个用户的“分数” (w1*benefit - w2*cost)
#         potential_scores = (self.w1 * self.demand * F_matrix) - (self.w2 * selected_dists)
#         potential_scores[selected_A == 0] = -np.inf  # 无覆盖的设为极小值
#
#         # 对每个用户，找到最高的分数
#         best_scores_per_user = np.max(potential_scores, axis=0)
#
#         covered_mask = best_scores_per_user > -np.inf
#         best_scores_per_user[~covered_mask] = 0  # 未覆盖的用户分数为0
#
#         # 适应度是所有用户最佳分数的总和
#         fitness_value = np.sum(best_scores_per_user)
#
#         # 计算覆盖率
#         total_demand = np.sum(self.demand)
#         covered_demand = np.sum(self.demand[covered_mask])
#         coverage_rate = covered_demand / total_demand if total_demand > 0 else 0
#
#         return fitness_value, coverage_rate
#
#     def initialize_population(self):
#         population = []
#         for _ in range(self.population_size):
#             chrom = random.sample(range(self.m), self.p)
#             fitness, coverage_rate = self.fitness(chrom)  # 修改接收的返回值
#             population.append({'chromosome': chrom, 'fitness': fitness, 'coverage_rate': coverage_rate})
#             if fitness > self.best_fitness:
#                 self.best_solution = chrom
#                 self.best_fitness = fitness
#                 self.best_coverage_rate = coverage_rate
#         return population
#
#     def selection(self, population):
#         sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
#         return sorted_pop[:self.elite_size]
#
#     def crossover(self, parent1, parent2):
#         # 使用基于集合的交叉，更适合无序的索引列表
#         p1_set = set(parent1)
#         p2_set = set(parent2)
#
#         common = list(p1_set & p2_set)
#         p1_only = list(p1_set - p2_set)
#         p2_only = list(p2_set - p1_set)
#
#         # 保证子代的多样性
#         child1_genes = common + random.sample(p1_only + p2_only, self.p - len(common))
#         child2_genes = common + random.sample(p1_only + p2_only, self.p - len(common))
#
#         random.shuffle(child1_genes)
#         random.shuffle(child2_genes)
#
#         return child1_genes, child2_genes
#
#     def mutate(self, chromosome):
#         if random.random() < self.mutation_prob:
#             idx_to_replace = random.randrange(self.p)
#             current_genes = set(chromosome)
#
#             # 从所有未被选择的设施中随机选一个
#             options = [i for i in range(self.m) if i not in current_genes]
#             if options:
#                 chromosome[idx_to_replace] = random.choice(options)
#         return chromosome
#
#     def optimize(self):
#         population = self.initialize_population()
#
#         # 打印初始最佳结果
#         print(f"Generation 0: Best Fitness = {self.best_fitness:.2f}, Coverage Rate = {self.best_coverage_rate:.2%}")
#
#         for gen in range(1, self.generations + 1):
#             next_population = self.selection(population)
#
#             while len(next_population) < self.population_size:
#                 # 使用轮盘赌或锦标赛选择可能效果更好，但随机抽样也可用
#                 parents = random.sample(population, 2)
#
#                 c1, c2 = self.crossover(parents[0]['chromosome'], parents[1]['chromosome'])
#                 c1 = self.mutate(c1)
#                 c2 = self.mutate(c2)
#
#                 fit1, cov1 = self.fitness(c1)
#                 fit2, cov2 = self.fitness(c2)
#
#                 next_population.append({'chromosome': c1, 'fitness': fit1, 'coverage_rate': cov1})
#                 if len(next_population) < self.population_size:
#                     next_population.append({'chromosome': c2, 'fitness': fit2, 'coverage_rate': cov2})
#
#                 if fit1 > self.best_fitness:
#                     self.best_solution = c1
#                     self.best_fitness = fit1
#                     self.best_coverage_rate = cov1
#                 if fit2 > self.best_fitness:
#                     self.best_solution = c2
#                     self.best_fitness = fit2
#                     self.best_coverage_rate = cov2
#
#             population = next_population
#
#             if gen % 10 == 0:  # 每10代打印一次进度
#                 print(
#                     f"Generation {gen}: Best Fitness = {self.best_fitness:.2f}, Coverage Rate = {self.best_coverage_rate:.2%}")
#
#         print("Optimization finished.")
#         return self.best_solution, self.best_fitness, self.best_coverage_rate

import random
import copy
import numpy as np

# (Keep your Chromosome and GeneticAlgorithmVectorized classes as they are)
# I am including them here for completeness.

class Chromosome:
    def __init__(self, content, fitness):
        self.content = content
        self.fitness = fitness

    def __str__(self): return "%s f=%.2f" % (self.content, self.fitness)
    def __repr__(self): return "%s f=%.2f" % (self.content, self.fitness)

class GeneticAlgorithmVectorized:
    def __init__(self, n, m, p, users, facilities, demand, w1, w2, da, db, D_max, A, dist,
                 population_size=100, generations=200, mutation_prob=0.2, elite_size=5):
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
        self.elite_size = elite_size

        self.best_solution = None
        self.best_fitness = -np.inf
        self.best_coverage = 0
        self.best_penalty = 0

    def F_vectorized(self, d):
        res = np.zeros_like(d)
        res[d <= self.da] = 1.0
        mask = (d > self.da) & (d <= self.db)
        # 确保分母不为零
        if self.db > self.da:
            res[mask] = 0.5 + 0.5 * np.cos(
                 (np.pi / (self.db - self.da)) * (d[mask] - (self.da + self.db) / 2) + np.pi / 2)
        return res

    def fitness(self, chromosome):
        """
               Calculates the fitness of a chromosome based on the objective function.
               A user is served by the single best facility that covers it.
               """
        selected_dists = self.dist_matrix[chromosome, :]
        selected_A = self.A[chromosome, :]

        F_matrix = self.F_vectorized(selected_dists)

        total_demand = np.sum(self.demand)

        # Old objective: w1*benefit - w2*cost
        # potential_scores = (self.w1 * self.demand * F_matrix) - (self.w2 * selected_dists)

        # New objective: normalized effective coverage - normalized demand-weighted distance burden
        potential_scores = (
            self.w1 * self.demand * F_matrix / total_demand
            - self.w2 * self.demand * selected_dists / (self.db * total_demand)
        )
        potential_scores[selected_A == 0] = -np.inf

        best_scores_per_user = np.max(potential_scores, axis=0)

        covered_mask = best_scores_per_user > -np.inf
        best_scores_per_user[~covered_mask] = 0

        # Add back the constant term w2
        fitness_value = np.sum(best_scores_per_user) + self.w2

        covered_demand = np.sum(self.demand[covered_mask])
        coverage_rate = covered_demand / total_demand if total_demand > 0 else 0

        return fitness_value, coverage_rate

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            chrom = random.sample(range(self.m), self.p)
            fitness, coverage_rate = self.fitness(chrom)  # 修改接收的返回值
            population.append({'chromosome': chrom, 'fitness': fitness, 'coverage_rate': coverage_rate})
            if fitness > self.best_fitness:
                self.best_solution = chrom
                self.best_fitness = fitness
                self.best_coverage_rate = coverage_rate
        return population

    def selection(self, population):
        sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
        return sorted_pop[:self.elite_size]

    def crossover(self, parent1, parent2):
        # 使用基于集合的交叉，更适合无序的索引列表
        p1_set = set(parent1)
        p2_set = set(parent2)

        common = list(p1_set & p2_set)
        p1_only = list(p1_set - p2_set)
        p2_only = list(p2_set - p1_set)

        # 保证子代的多样性
        child1_genes = common + random.sample(p1_only + p2_only, self.p - len(common))
        child2_genes = common + random.sample(p1_only + p2_only, self.p - len(common))

        random.shuffle(child1_genes)
        random.shuffle(child2_genes)

        return child1_genes, child2_genes

    def mutate(self, chromosome):
        if random.random() < self.mutation_prob:
            idx_to_replace = random.randrange(self.p)
            current_genes = set(chromosome)

            # 从所有未被选择的设施中随机选一个
            options = [i for i in range(self.m) if i not in current_genes]
            if options:
                chromosome[idx_to_replace] = random.choice(options)
        return chromosome

    def optimize(self):
        population = self.initialize_population()

        # Suppressing the print output from within the class for cleaner looping
        # print(f"Generation 0: Best Fitness = {self.best_fitness:.2f}, Coverage Rate = {self.best_coverage_rate:.2%}")

        for gen in range(1, self.generations + 1):
            next_population = self.selection(population)

            while len(next_population) < self.population_size:
                parents = random.sample(population, 2)
                c1, c2 = self.crossover(parents[0]['chromosome'], parents[1]['chromosome'])
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                fit1, cov1 = self.fitness(c1)
                fit2, cov2 = self.fitness(c2)

                next_population.append({'chromosome': c1, 'fitness': fit1, 'coverage_rate': cov1})
                if len(next_population) < self.population_size:
                    next_population.append({'chromosome': c2, 'fitness': fit2, 'coverage_rate': cov2})

                if fit1 > self.best_fitness:
                    self.best_solution = c1
                    self.best_fitness = fit1
                    self.best_coverage_rate = cov1
                if fit2 > self.best_fitness:
                    self.best_solution = c2
                    self.best_fitness = fit2
                    self.best_coverage_rate = cov2
            population = next_population
            # Suppressing generation prints for cleaner looping
            # if gen % 10 == 0:
            #     print(f"Generation {gen}: Best Fitness = {self.best_fitness:.2f}, Coverage Rate = {self.best_coverage_rate:.2%}")

        # print("Optimization finished.")
        return self.best_solution, self.best_fitness, self.best_coverage_rate