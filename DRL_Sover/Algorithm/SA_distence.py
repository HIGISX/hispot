# import numpy as np
# import random
#
# class SimulatedAnnealingVectorized:
#     def __init__(self, n, m, p, users, facilities, demand, w1, w2, da, db, D_max, A, dist,
#                  T0=1000, alpha=0.95, stopping_T=1e-3, max_iter=1000):
#         self.n = n
#         self.m = m
#         self.p = p
#         self.users = users
#         self.facilities = facilities
#         self.demand = demand  # shape (n,)
#         self.w1 = w1
#         self.w2 = w2
#         self.da = da
#         self.db = db
#         self.D_max = D_max
#         self.A = A  # shape (m, n), coverage matrix
#         self.dist_matrix = dist  # shape (m, n), precomputed distances
#
#         # SA parameters
#         self.T = T0
#         self.alpha = alpha
#         self.stopping_T = stopping_T
#         self.max_iter = max_iter
#
#         self.best_solution = None
#         self.best_fitness = -np.inf
#         self.best_coverage = 0
#         self.best_penalty = 0
#
#     def F_vectorized(self, d):
#         res = np.zeros_like(d)
#         res[d <= self.da] = 1
#         mask = (d > self.da) & (d <= self.db)
#         res[mask] = 0.5 * np.cos((np.pi / (self.db - self.da)) * (d[mask] - (self.da + self.db) / 2) + np.pi / 2)
#         return res
#
#     def fitness(self, chromosome):
#         selected_dists = self.dist_matrix[chromosome, :]  # shape (p, n)
#         selected_A = self.A[chromosome, :]  # shape (p, n)
#
#         covered_mask = selected_A.any(axis=0)  # shape (n,)
#         F_matrix = self.F_vectorized(selected_dists)
#
#         benefit = (F_matrix * selected_A) @ self.demand
#         benefit = benefit.sum()
#
#         distance_penalty = (selected_dists * selected_A).sum()
#         uncovered_penalty = (~covered_mask).sum() * 1000
#
#         total_demand = self.demand.sum()
#         covered_demand = self.demand[covered_mask].sum()
#         coverage_rate = covered_demand / total_demand if total_demand > 0 else 0
#
#         fitness_value = self.w1 * benefit - self.w2 * distance_penalty
#
#         return fitness_value, coverage_rate, uncovered_penalty
#
#     def neighbor(self, current):
#         neighbor = current.copy()
#         idx = random.randint(0, self.p - 1)
#         available = [x for x in range(self.m) if x not in neighbor]
#         if available:
#             neighbor[idx] = random.choice(available)
#         return neighbor
#
#     def optimize(self):
#         current = random.sample(range(self.m), self.p)
#         current_fitness, current_cov, current_penalty = self.fitness(current)
#         self.best_solution = current
#         self.best_fitness = current_fitness
#         self.best_coverage = current_cov
#         self.best_penalty = current_penalty
#
#         iter_count = 0
#         while self.T > self.stopping_T and iter_count < self.max_iter:
#             candidate = self.neighbor(current)
#             cand_fitness, cand_cov, cand_penalty = self.fitness(candidate)
#
#             if cand_fitness > current_fitness:
#                 current, current_fitness = candidate, cand_fitness
#             else:
#                 prob = np.exp((cand_fitness - current_fitness) / self.T)
#                 if random.random() < prob:
#                     current, current_fitness = candidate, cand_fitness
#
#             if current_fitness > self.best_fitness:
#                 self.best_solution = candidate
#                 self.best_fitness = cand_fitness
#                 self.best_coverage = cand_cov
#                 self.best_penalty = cand_penalty
#
#             self.T *= self.alpha
#             iter_count += 1
#
#         return self.best_solution, self.best_fitness, self.best_coverage, self.best_penalty
import numpy as np
import random


class SimulatedAnnealingVectorized:
    def __init__(self, n, m, p, users, facilities, demand, w1, w2, da, db, D_max, A, dist,
                 T0=1000, alpha=0.95, stopping_T=1e-3, max_iter=1000):
        self.n = n
        self.m = m
        self.p = p
        self.users = users
        self.facilities = facilities
        self.demand = demand  # 形状为 (n,) 的需求向量
        self.w1 = w1
        self.w2 = w2
        self.da = da
        self.db = db
        self.D_max = D_max
        self.A = A  # 形状为 (m, n) 的覆盖矩阵
        self.dist_matrix = dist  # 形状为 (m, n) 的预计算距离矩阵

        # 模拟退火参数
        self.T = T0
        self.alpha = alpha
        self.stopping_T = stopping_T
        self.max_iter = max_iter

        self.best_solution = None
        self.best_fitness = -np.inf
        self.best_coverage_rate = 0  # 与GA保持一致

    def F_vectorized(self, d):
        """
        以向量化方式计算距离衰减函数 F(d)。
        此函数已更新，以匹配GA的实现。
        """
        res = np.zeros_like(d, dtype=float)
        res[d <= self.da] = 1.0
        mask = (d > self.da) & (d <= self.db)
        # 确保分母不为零
        if self.db > self.da:
            # res[mask] = 0.5 + 0.5 * np.cos((np.pi / (self.db - self.da)) * (d[mask] - self.db))
            res[mask] = 0.5 + 0.5 * np.cos(
                (np.pi / (self.db - self.da)) * (d[mask] - (self.da + self.db) / 2) + np.pi / 2)
        return res

    def fitness(self, chromosome):
        """
        根据目标函数计算染色体的适应度，逻辑与 GeneticAlgorithmVectorized 类中的相匹配。
        一个用户由覆盖它的、能提供最高单一分值的设施服务。
        """
        # 获取所选设施到所有用户的距离和覆盖情况
        selected_dists = self.dist_matrix[chromosome, :]  # 形状 (p, n)
        selected_A = self.A[chromosome, :]  # 形状 (p, n)

        # 计算所有选中设施对所有用户的距离衰减因子
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

    def get_neighbor(self, current_solution):
        """
        通过随机替换一个已选设施来生成一个邻域解。
        """
        neighbor = current_solution.copy()
        # 随机选择一个在当前解中要被替换的设施的索引
        idx_to_replace = random.randint(0, self.p - 1)

        # 当前解中已包含的设施集合
        current_genes = set(neighbor)

        # 候选设施是所有未被选中的设施
        available_facilities = [i for i in range(self.m) if i not in current_genes]

        if available_facilities:
            # 从候选设施中随机选择一个新设施进行替换
            new_facility = random.choice(available_facilities)
            neighbor[idx_to_replace] = new_facility

        return neighbor

    def optimize(self):
        """
        运行模拟退火优化过程。
        该逻辑已更新，以模仿GA的初始化和结果跟踪方式。
        """
        # 1. 初始化
        # 生成一个随机的初始解
        current_solution = random.sample(range(self.m), self.p)
        # 计算初始解的适应度和覆盖率
        current_fitness, current_coverage = self.fitness(current_solution)

        # 用初始解初始化迄今为止找到的最优解
        self.best_solution = current_solution
        self.best_fitness = current_fitness
        self.best_coverage_rate = current_coverage

        print(f"初始状态: 最优适应度 = {self.best_fitness:.2f}, 覆盖率 = {self.best_coverage_rate:.2%}")

        iter_count = 0
        # 2. 主循环
        while self.T > self.stopping_T and iter_count < self.max_iter:
            # 生成一个邻域解
            candidate_solution = self.get_neighbor(current_solution)
            # 计算其适应度
            candidate_fitness, candidate_coverage = self.fitness(candidate_solution)

            # 计算适应度的变化量
            delta_fitness = candidate_fitness - current_fitness

            # 3. 接受准则
            # 如果新解更优，则总是接受
            if delta_fitness > 0:
                current_solution = candidate_solution
                current_fitness = candidate_fitness
                current_coverage = candidate_coverage
            # 如果新解更差，则以一定概率接受（Metropolis准则）
            else:
                # 计算接受概率
                acceptance_probability = np.exp(delta_fitness / self.T)
                if random.random() < acceptance_probability:
                    current_solution = candidate_solution
                    current_fitness = candidate_fitness
                    current_coverage = candidate_coverage

            # 4. 更新最优解
            # 检查当前解是否是迄今为止找到的最好的解
            if current_fitness > self.best_fitness:
                self.best_solution = current_solution
                self.best_fitness = current_fitness
                self.best_coverage_rate = current_coverage

            # 5. 降温
            self.T *= self.alpha
            iter_count += 1

            if iter_count % 100 == 0:  # 每100次迭代打印一次进度
                print(
                    f"迭代次数 {iter_count}: 温度 = {self.T:.2f}, 当前适应度 = {current_fitness:.2f}, 最优适应度 = {self.best_fitness:.2f}")

        print("优化完成。")
        return self.best_solution, self.best_fitness, self.best_coverage_rate