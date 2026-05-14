import numpy as np
import random
import copy


class ParticleSwarmOptimization:
    def __init__(self, n, m, p, users, facilities, demand, w1, w2, da, db, D_max, A, dist,
                 num_particles=50, max_iter=200, inertia=0.7, cognitive_coeff=1.5, social_coeff=1.5):
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

        # PSO Parameters
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.inertia = inertia
        self.c1 = cognitive_coeff
        self.c2 = social_coeff

        # Swarm state
        self.particles_pos = []
        self.particles_vel = []
        self.pbest_pos = []
        self.pbest_fitness = []

        self.gbest_pos = None
        self.gbest_fitness = -np.inf
        self.gbest_coverage_rate = 0

    def F_vectorized(self, d):
        """
        Vectorized distance decay function F(d).
        THIS FUNCTION HAS BEEN CORRECTED.
        """
        res = np.zeros_like(d, dtype=float)
        # Condition 1: d <= da
        res[d <= self.da] = 1.0
        # Condition 2: da < d <= db
        mask = (d > self.da) & (d <= self.db)
        if self.db > self.da:
            # CORRECTED a bug in the formula to match Gurobi and GA logic.
            # It now correctly decays from 1 to 0.
            res[mask] = 0.5 + 0.5 * np.cos(
                (np.pi / (self.db - self.da)) * (d[mask] - (self.da + self.db) / 2) + np.pi / 2)
        # Condition 3: d > db -> res remains 0
        return res

    def fitness(self, chromosome):
        """
        Calculates the fitness of a solution (chromosome/particle position).
        This function is identical to the one in the GA and SA vectorized versions.
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

    def _initialize_swarm(self):
        """
        Initializes the swarm's positions, velocities, and bests.
        """
        for _ in range(self.num_particles):
            pos = random.sample(range(self.m), self.p)
            self.particles_pos.append(pos)
            self.particles_vel.append([])

            fitness, coverage = self.fitness(pos)
            self.pbest_pos.append(pos)
            self.pbest_fitness.append(fitness)

            if fitness > self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_pos = pos
                self.gbest_coverage_rate = coverage

        print(
            f"Initialization: Global Best Fitness = {self.gbest_fitness:.2f}, Coverage = {self.gbest_coverage_rate:.2%}")

    def _update_velocity(self, i):
        """
        Update the velocity for particle 'i' based on discrete PSO logic.
        """
        pos_current = set(self.particles_pos[i])
        pos_pbest = set(self.pbest_pos[i])
        pos_gbest = set(self.gbest_pos)

        vel_new = []

        if random.random() < self.inertia:
            vel_new.extend(self.particles_vel[i])

        r1 = random.random()
        swaps_to_pbest = (pos_pbest - pos_current)
        if swaps_to_pbest:
            for _ in range(int(self.c1 * r1 * len(swaps_to_pbest))):
                if (pos_current - pos_pbest):
                    to_add = random.choice(list(swaps_to_pbest))
                    to_remove = random.choice(list(pos_current - pos_pbest))
                    vel_new.append((to_add, to_remove))

        r2 = random.random()
        swaps_to_gbest = (pos_gbest - pos_current)
        if swaps_to_gbest:
            for _ in range(int(self.c2 * r2 * len(swaps_to_gbest))):
                if (pos_current - pos_gbest):
                    to_add = random.choice(list(swaps_to_gbest))
                    to_remove = random.choice(list(pos_current - pos_gbest))
                    vel_new.append((to_add, to_remove))

        self.particles_vel[i] = vel_new

    def _update_position(self, i):
        """
        Update the position of particle 'i' by applying its velocity (swaps).
        """
        pos = self.particles_pos[i]
        vel = self.particles_vel[i]

        pos_set = set(pos)

        if not vel:
            return

        for to_add, to_remove in vel:
            if to_remove in pos_set and to_add not in pos_set:
                pos_set.remove(to_remove)
                pos_set.add(to_add)

        self.particles_pos[i] = list(pos_set)

    def optimize(self):
        """
        Run the Particle Swarm Optimization algorithm.
        """
        self._initialize_swarm()

        for iter_num in range(1, self.max_iter + 1):
            for i in range(self.num_particles):
                self._update_velocity(i)
                self._update_position(i)

                pos_new = self.particles_pos[i]
                fitness_new, coverage_new = self.fitness(pos_new)

                if fitness_new > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness_new
                    self.pbest_pos[i] = pos_new

                if fitness_new > self.gbest_fitness:
                    self.gbest_fitness = fitness_new
                    self.gbest_pos = pos_new
                    self.gbest_coverage_rate = coverage_new

            if iter_num % 10 == 0:
                print(
                    f"Iteration {iter_num}: Global Best Fitness = {self.gbest_fitness:.2f}, Coverage = {self.gbest_coverage_rate:.2%}")

        print("\nOptimization finished.")
        return self.gbest_pos, self.gbest_fitness, self.gbest_coverage_rate