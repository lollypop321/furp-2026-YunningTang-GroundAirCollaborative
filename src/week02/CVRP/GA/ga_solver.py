import random
import numpy as np
from copy import deepcopy


class GeneticAlgorithm:
    """
    遗传算法求解 CVRP with EV and TW
    """
    def __init__(self, env, pop_size=100, cx_pb=0.85, mut_pb=0.02, n_gen=200,
                 use_elitism=True, elitism_rate=0.05):
        self.env = env
        self.pop_size = pop_size
        self.cx_pb = cx_pb
        self.mut_pb = mut_pb
        self.n_gen = n_gen
        self.use_elitism = use_elitism
        self.elitism_rate = elitism_rate

        self.problem_size = env.problem_size
        self.best_individual = None
        self.best_cost = float('inf')
        self.history = []

    def create_individual(self):
        """创建随机个体（节点排列）"""
        return random.sample(range(1, self.problem_size + 1), self.problem_size)

    def create_population(self):
        """创建初始种群"""
        return [self.create_individual() for _ in range(self.pop_size)]

    def evaluate_population(self, population):
        """评估种群"""
        costs = []
        for ind in population:
            cost = self.env.evaluate_individual(ind)
            costs.append(cost)
        return costs

    def select_parent(self, population, costs):
        """锦标赛选择"""
        tournament_size = 3
        tournament = random.sample(list(zip(population, costs)), tournament_size)
        tournament.sort(key=lambda x: x[1])
        return tournament[0][0]

    def cx_partially_matched(self, ind1, ind2):
        """部分匹配交叉 (PMX) - 简化版"""
        size = len(ind1)
        cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))

        # 创建子代（深拷贝）
        child1 = ind1[:]
        child2 = ind2[:]

        # 交换交叉区域
        child1[cxpoint1:cxpoint2+1], child2[cxpoint1:cxpoint2+1] = \
            child2[cxpoint1:cxpoint2+1], child1[cxpoint1:cxpoint2+1]

        # 修复重复基因
        # 对于 child1，交叉区域来自 ind2，需要修复 ind1 区域的重复
        # 对于 child2，交叉区域来自 ind1，需要修复 ind2 区域的重复

        # 修复 child1
        segment1 = set(child1[cxpoint1:cxpoint2+1])  # 来自 ind2
        for i in list(range(cxpoint1)) + list(range(cxpoint2+1, size)):
            if child1[i] in segment1:
                # 找到映射
                val = child1[i]
                while val in segment1:
                    # 在 ind2 的对应位置找到映射
                    idx = ind2.index(val)
                    val = ind1[idx]
                child1[i] = val

        # 修复 child2
        segment2 = set(child2[cxpoint1:cxpoint2+1])  # 来自 ind1
        for i in list(range(cxpoint1)) + list(range(cxpoint2+1, size)):
            if child2[i] in segment2:
                val = child2[i]
                while val in segment2:
                    idx = ind1.index(val)
                    val = ind2[idx]
                child2[i] = val

        return child1, child2

    def mut_inverse(self, individual):
        """逆序变异"""
        size = len(individual)
        start, stop = sorted(random.sample(range(size), 2))
        individual[start:stop+1] = reversed(individual[start:stop+1])
        return individual

    def mut_swap(self, individual):
        """交换变异"""
        size = len(individual)
        idx1, idx2 = random.sample(range(size), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def run(self, verbose=True):
        """运行遗传算法"""
        # 初始化种群
        population = self.create_population()
        costs = self.evaluate_population(population)

        # 更新最优解
        min_cost = min(costs)
        min_idx = costs.index(min_cost)
        if min_cost < self.best_cost:
            self.best_cost = min_cost
            self.best_individual = deepcopy(population[min_idx])

        self.history.append(min_cost)

        if verbose:
            print(f"Initial best cost: {min_cost:.4f}")

        # 进化循环
        for gen in range(self.n_gen):
            new_population = []

            # 精英保留
            if self.use_elitism:
                n_elite = max(1, int(self.pop_size * self.elitism_rate))
                sorted_indices = np.argsort(costs)
                elites = [deepcopy(population[i]) for i in sorted_indices[:n_elite]]
                new_population.extend(elites)

            # 生成新个体
            while len(new_population) < self.pop_size:
                # 选择父代
                parent1 = self.select_parent(population, costs)
                parent2 = self.select_parent(population, costs)

                # 交叉
                if random.random() < self.cx_pb:
                    child1, child2 = self.cx_partially_matched(deepcopy(parent1), deepcopy(parent2))
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)

                # 变异
                if random.random() < self.mut_pb:
                    child1 = self.mut_inverse(child1)
                if random.random() < self.mut_pb:
                    child2 = self.mut_swap(child2)

                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)

            # 更新种群
            population = new_population[:self.pop_size]
            costs = self.evaluate_population(population)

            # 更新最优解
            min_cost = min(costs)
            min_idx = costs.index(min_cost)
            if min_cost < self.best_cost:
                self.best_cost = min_cost
                self.best_individual = deepcopy(population[min_idx])

            self.history.append(self.best_cost)

            if verbose and (gen + 1) % 20 == 0:
                avg_cost = sum(costs) / len(costs)
                print(f"Gen {gen+1}/{self.n_gen}: Best={self.best_cost:.4f}, Avg={avg_cost:.4f}")

        if verbose:
            print(f"\nFinal best cost: {self.best_cost:.4f}")

        return self.best_individual, self.best_cost
