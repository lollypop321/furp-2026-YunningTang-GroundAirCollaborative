import torch
import numpy as np
import random
from copy import deepcopy


class CVRPGAEnv:
    """
    CVRP 环境，支持 EV 能量约束和时间窗约束
    与 POMO 版本保持一致
    """
    def __init__(self, problem_data, use_ev=True, use_tw=True):
        self.problem_data = problem_data
        self.use_ev = use_ev
        self.use_tw = use_tw

        # 提取数据
        self.depot_xy = problem_data['depot_xy'].squeeze(0).numpy()  # (2,)
        self.node_xy = problem_data['node_xy'].squeeze(0).numpy()    # (n, 2)
        self.node_demand = problem_data['node_demand'].squeeze(0).numpy()  # (n,)

        self.problem_size = self.node_xy.shape[0]
        self.vehicle_capacity = 1.0  # 标准化容量

        if use_tw:
            self.node_service_time = problem_data['node_service_time'].squeeze(0).numpy()
            self.tw_early = problem_data['tw_early'].squeeze(0).numpy()
            self.tw_late = problem_data['tw_late'].squeeze(0).numpy()

        if use_ev:
            self.battery_capacity = problem_data['battery_capacity']
            self.energy_consumption_rate = problem_data['energy_consumption_rate']
            self.initial_battery = problem_data['initial_battery'].squeeze(0).item()

        # 预计算距离矩阵
        all_nodes = np.vstack([self.depot_xy.reshape(1, -1), self.node_xy])
        self.distance_matrix = np.sqrt(((all_nodes[:, None, :] - all_nodes[None, :, :]) ** 2).sum(axis=2))

    def decode_individual(self, individual):
        """
        将个体（节点排列）解码为实际路径
        考虑容量、时间窗、能量约束
        """
        routes = []
        unvisited = set(individual)

        while unvisited:
            route = []
            current_load = 0.0
            current_time = 0.0
            current_battery = self.initial_battery if self.use_ev else None
            current_node = 0  # depot

            for customer_id in list(unvisited):
                demand = self.node_demand[customer_id - 1]

                # 检查容量约束
                if current_load + demand > self.vehicle_capacity:
                    continue

                # 计算到该客户的距离和时间
                dist = self.distance_matrix[current_node, customer_id]
                arrival_time = current_time + dist

                # 检查时间窗约束
                if self.use_tw:
                    service_time = self.node_service_time[customer_id - 1]
                    tw_early = self.tw_early[customer_id - 1]
                    tw_late = self.tw_late[customer_id - 1]

                    # 等待到 tw_early
                    start_time = max(arrival_time, tw_early)

                    # 检查是否能在 tw_late 前完成服务
                    if start_time > tw_late:
                        continue

                    # 计算离开时间
                    leave_time = start_time + service_time
                else:
                    leave_time = arrival_time

                # 检查能量约束
                if self.use_ev:
                    energy_needed = dist * self.energy_consumption_rate
                    return_dist = self.distance_matrix[customer_id, 0]
                    return_energy = return_dist * self.energy_consumption_rate

                    # 需要足够电量到达客户并返回 depot
                    if current_battery < energy_needed + return_energy:
                        continue

                    # 更新电量（在客户处不充电，只在 depot 充电）
                    new_battery = current_battery - energy_needed
                else:
                    new_battery = None

                # 所有约束满足，加入路径
                route.append(customer_id)
                unvisited.remove(customer_id)
                current_load += demand
                current_time = leave_time
                current_node = customer_id
                if self.use_ev:
                    current_battery = new_battery

            # 结束当前路径
            if route:
                routes.append(route)
            else:
                # 无法服务任何剩余客户，强制开启新路径（从depot出发）
                # 这种情况不应该发生，但如果发生，强制服务第一个客户
                first_customer = list(unvisited)[0]
                routes.append([first_customer])
                unvisited.remove(first_customer)

        return routes

    def calculate_route_cost(self, route):
        """计算单条路径的总距离"""
        if not route:
            return 0.0

        total_dist = 0.0
        current_node = 0  # depot

        for customer_id in route:
            total_dist += self.distance_matrix[current_node, customer_id]
            current_node = customer_id

        # 返回 depot
        total_dist += self.distance_matrix[current_node, 0]

        return total_dist

    def evaluate_individual(self, individual):
        """
        评估个体（适应度函数）
        返回总行驶距离，越小越好
        """
        routes = self.decode_individual(individual)

        if not routes:
            return float('inf')  # 无效解

        total_cost = 0.0
        for route in routes:
            total_cost += self.calculate_route_cost(route)

        return total_cost

    def count_violations(self, individual):
        """统计约束违反情况（用于分析）"""
        routes = self.decode_individual(individual)

        tw_violations = 0
        energy_violations = 0

        for route in routes:
            current_time = 0.0
            current_battery = self.initial_battery if self.use_ev else None
            current_node = 0

            for customer_id in route:
                dist = self.distance_matrix[current_node, customer_id]
                arrival_time = current_time + dist

                if self.use_tw:
                    tw_late = self.tw_late[customer_id - 1]
                    if arrival_time > tw_late:
                        tw_violations += 1

                    service_time = self.node_service_time[customer_id - 1]
                    tw_early = self.tw_early[customer_id - 1]
                    current_time = max(arrival_time, tw_early) + service_time
                else:
                    current_time = arrival_time

                if self.use_ev:
                    energy_needed = dist * self.energy_consumption_rate
                    current_battery -= energy_needed
                    if current_battery < 0:
                        energy_violations += 1

                current_node = customer_id

        return tw_violations, energy_violations
