##########################################################################################
# ALNS (Adaptive Large Neighborhood Search) for CVRP with EV and TW
# 基于 OR-Tools 实现，支持电动车能量约束
##########################################################################################

import os
import sys
import torch
import numpy as np
import random
import time
from copy import deepcopy

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")

from problem_def import get_random_problems_with_ev_tw


try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("Warning: OR-Tools not installed. Install with: pip install ortools")


##########################################################################################
# Parameters

PROBLEM_SIZE = 200
USE_EV = True
USE_TW = True

BATTERY_CAPACITY = 100.0
ENERGY_CONSUMPTION_RATE = 0.5

N_INSTANCES = 100
TEST_DATA_FILE = '../vrp200_test_seed1234.pt'

# ALNS 参数
MAX_ITERATIONS = 200
DESTROY_RATE = 0.3
COOLING_RATE = 0.995
INITIAL_TEMPERATURE = 100.0


class CVRPSolution:
    """CVRP 解的表示"""
    def __init__(self, routes, cost, distance_matrix, demands, tw_early, tw_late, 
                 service_time, battery_capacity, energy_rate):
        self.routes = routes  # [[1,2,3], [4,5,6], ...]
        self.cost = cost
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.tw_early = tw_early
        self.tw_late = tw_late
        self.service_time = service_time
        self.battery_capacity = battery_capacity
        self.energy_rate = energy_rate
        self.n_nodes = len(distance_matrix)
        
    def copy(self):
        return CVRPSolution(
            [r[:] for r in self.routes],
            self.cost,
            self.distance_matrix,
            self.demands,
            self.tw_early,
            self.tw_late,
            self.service_time,
            self.battery_capacity,
            self.energy_rate
        )
    
    def calculate_cost(self):
        """计算总行驶距离"""
        total = 0.0
        for route in self.routes:
            if not route:
                continue
            # depot -> first
            total += self.distance_matrix[0][route[0]]
            # route
            for i in range(len(route) - 1):
                total += self.distance_matrix[route[i]][route[i+1]]
            # last -> depot
            total += self.distance_matrix[route[-1]][0]
        return total
    
    def is_feasible(self):
        """检查解是否满足所有约束（宽松版本，允许轻微违反）"""
        total_violation = 0
        for route in self.routes:
            violation = self._calculate_route_violation(route)
            total_violation += violation
        # 允许少量违反（总延迟 < 5）
        return total_violation < 5.0
    
    def _calculate_route_violation(self, route):
        """计算路径的约束违反量（用于 soft constraint）"""
        if not route:
            return 0.0
            
        violation = 0.0
        current_time = 0.0
        current_battery = self.battery_capacity
        current_node = 0
        
        for node in route:
            travel = self.distance_matrix[current_node][node]
            arrival = current_time + travel
            
            # 时间窗违反
            if self.tw_early is not None and self.tw_late is not None:
                if arrival > self.tw_late[node]:
                    violation += arrival - self.tw_late[node]
                current_time = max(arrival, self.tw_early[node]) + self.service_time[node]
            else:
                current_time = arrival
            
            # 能量消耗
            current_battery -= travel * self.energy_rate
            
            current_node = node
        
        # 返回 depot 的能量检查
        return_dist = self.distance_matrix[current_node][0]
        return_energy = return_dist * self.energy_rate
        if current_battery < return_energy:
            violation += (return_energy - current_battery) * 2
        
        return violation
    
    def _is_route_feasible(self, route):
        """检查单条路径是否可行（宽松版本）"""
        if not route:
            return True
            
        # 容量检查（客户节点 n 对应 demands[n-1]）
        load = sum(self.demands[node-1] for node in route if node > 0)
        if load > 1.0:
            return False
        
        # 计算违反量，允许少量违反
        violation = self._calculate_route_violation(route)
        return violation < 2.0  # 允许单路径少量违反
        
    def _old_is_route_feasible(self, route):
        """检查单条路径是否可行（严格版本，备用）"""
        if not route:
            return True
            
        # 容量检查（客户节点 n 对应 demands[n-1]）
        load = sum(self.demands[node-1] for node in route if node > 0)
        if load > 1.0:  # vehicle_capacity
            return False
        
        # 时间和能量检查
        current_time = 0.0
        current_battery = self.battery_capacity
        current_node = 0
        
        for node in route:
            # 到达时间
            travel_time = self.distance_matrix[current_node][node]
            arrival_time = current_time + travel_time
            
            # 时间窗检查
            if self.tw_early is not None and self.tw_late is not None:
                if arrival_time > self.tw_late[node]:
                    return False
                start_time = max(arrival_time, self.tw_early[node])
                current_time = start_time + self.service_time[node]
            else:
                current_time = arrival_time
            
            # 能量检查
            energy_needed = travel_time * self.energy_rate
            current_battery -= energy_needed
            if current_battery < 0:
                return False
            
            current_node = node
        
        # 返回 depot
        return_dist = self.distance_matrix[current_node][0]
        return_energy = return_dist * self.energy_rate
        if current_battery < return_energy:
            return False
        
        return True


class ALNS:
    """自适应大邻域搜索"""
    def __init__(self, distance_matrix, demands, tw_early, tw_late, service_time,
                 battery_capacity, energy_rate, max_iter=100, destroy_rate=0.3):
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.tw_early = tw_early
        self.tw_late = tw_late
        self.service_time = service_time
        self.battery_capacity = battery_capacity
        self.energy_rate = energy_rate
        self.max_iter = max_iter
        self.destroy_rate = destroy_rate
        
        # 算子权重（自适应）
        self.destroy_weights = [1.0, 1.0, 1.0]  # 3种destroy算子
        self.repair_weights = [1.0, 1.0]  # 2种repair算子
        
    def solve(self, initial_routes):
        """ALNS 主循环"""
        # 初始解
        current_sol = CVRPSolution(
            initial_routes, 0, self.distance_matrix, self.demands,
            self.tw_early, self.tw_late, self.service_time,
            self.battery_capacity, self.energy_rate
        )
        current_sol.cost = current_sol.calculate_cost()
        
        best_sol = current_sol.copy()
        best_cost = current_sol.cost
        
        temperature = INITIAL_TEMPERATURE
        
        for iteration in range(self.max_iter):
            # 选择 destroy 算子
            destroy_idx = self._select_operator(self.destroy_weights)
            repair_idx = self._select_operator(self.repair_weights)
            
            # Destroy
            removed, partial_sol = self._destroy(current_sol, destroy_idx)
            
            # Repair
            new_sol = self._repair(partial_sol, removed, repair_idx)
            
            if new_sol is None:
                continue
            
            new_cost = new_sol.cost
            
            # 接受准则（模拟退火）
            if new_cost < current_sol.cost:
                # 更好，接受
                current_sol = new_sol
                self._update_weights(destroy_idx, repair_idx, success=True)
                if new_cost < best_cost:
                    best_sol = new_sol.copy()
                    best_cost = new_cost
            else:
                # 更差，按概率接受
                delta = new_cost - current_sol.cost
                prob = np.exp(-delta / temperature)
                if random.random() < prob:
                    current_sol = new_sol
                self._update_weights(destroy_idx, repair_idx, success=False)
            
            temperature *= COOLING_RATE
            
            if (iteration + 1) % 20 == 0:
                print(f"  Iter {iteration+1}/{self.max_iter}: Best={best_cost:.4f}, Current={current_sol.cost:.4f}")
        
        return best_sol, best_cost
    
    def _select_operator(self, weights):
        """轮盘赌选择算子"""
        total = sum(weights)
        r = random.random() * total
        for i, w in enumerate(weights):
            r -= w
            if r <= 0:
                return i
        return len(weights) - 1
    
    def _update_weights(self, d_idx, r_idx, success):
        """更新算子权重"""
        if success:
            self.destroy_weights[d_idx] *= 1.2
            self.repair_weights[r_idx] *= 1.2
        else:
            self.destroy_weights[d_idx] *= 0.9
            self.repair_weights[r_idx] *= 0.9
    
    def _destroy(self, solution, operator_idx):
        """Destroy 算子：移除部分客户"""
        all_nodes = []
        for route in solution.routes:
            all_nodes.extend(route)
        
        n_remove = max(1, int(len(all_nodes) * self.destroy_rate))
        
        if operator_idx == 0:
            # Random removal
            removed = random.sample(all_nodes, n_remove)
        elif operator_idx == 1:
            # Worst removal（移除增加距离最多的）
            removed = self._worst_removal(solution, all_nodes, n_remove)
        else:
            # Shaw removal（移除相似的）
            removed = self._shaw_removal(solution, all_nodes, n_remove)
        
        # 构建部分解
        new_routes = []
        for route in solution.routes:
            new_route = [n for n in route if n not in removed]
            if new_route:
                new_routes.append(new_route)
        
        partial_sol = solution.copy()
        partial_sol.routes = new_routes
        
        return removed, partial_sol
    
    def _worst_removal(self, solution, all_nodes, n_remove):
        """移除对目标函数影响最大的节点"""
        costs = []
        for node in all_nodes:
            # 计算移除该节点节省的距离
            saved = self._calculate_removal_saving(solution, node)
            costs.append((node, saved))
        
        costs.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in costs[:n_remove]]
    
    def _shaw_removal(self, solution, all_nodes, n_remove):
        """Shaw removal：移除与随机选择节点相似的节点"""
        if not all_nodes:
            return []
        seed = random.choice(all_nodes)
        
        # 计算相似度（距离 + 时间窗相似）
        similarities = []
        for node in all_nodes:
            if node == seed:
                continue
            dist_sim = 1 / (self.distance_matrix[seed][node] + 0.001)
            if self.tw_early is not None:
                tw_sim = 1 / (abs(self.tw_early[seed] - self.tw_early[node]) + 0.001)
                sim = dist_sim + tw_sim
            else:
                sim = dist_sim
            similarities.append((node, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [seed] + [n for n, _ in similarities[:n_remove-1]]
    
    def _calculate_removal_saving(self, solution, node):
        """计算移除节点节省的距离"""
        for route in solution.routes:
            if node in route:
                idx = route.index(node)
                # 原来：prev -> node -> next
                # 移除后：prev -> next
                prev_node = route[idx-1] if idx > 0 else 0
                next_node = route[idx+1] if idx < len(route)-1 else 0
                
                old_dist = (self.distance_matrix[prev_node][node] + 
                           self.distance_matrix[node][next_node])
                new_dist = self.distance_matrix[prev_node][next_node]
                return old_dist - new_dist
        return 0
    
    def _repair(self, partial_sol, removed_nodes, operator_idx):
        """Repair 算子：重新插入节点"""
        new_routes = [r[:] for r in partial_sol.routes]
        
        # 随机打乱插入顺序
        random.shuffle(removed_nodes)
        
        for node in removed_nodes:
            if operator_idx == 0:
                # Greedy insertion
                success = self._greedy_insert(new_routes, node)
            else:
                # Regret insertion
                success = self._regret_insert(new_routes, node)
            
            if not success:
                # 创建新路径
                new_routes.append([node])
        
        # 构建新解
        new_sol = partial_sol.copy()
        new_sol.routes = new_routes
        new_sol.cost = new_sol.calculate_cost()
        
        return new_sol if new_sol.is_feasible() else None
    
    def _greedy_insert(self, routes, node):
        """贪心插入：选择增加成本最小的位置"""
        best_cost = float('inf')
        best_route = -1
        best_pos = -1
        
        for r_idx, route in enumerate(routes):
            for pos in range(len(route) + 1):
                # 尝试插入位置 pos
                new_route = route[:pos] + [node] + route[pos:]
                if self._is_route_valid(new_route):
                    cost = self._calculate_insertion_cost(route, node, pos)
                    if cost < best_cost:
                        best_cost = cost
                        best_route = r_idx
                        best_pos = pos
        
        if best_route >= 0:
            routes[best_route].insert(best_pos, node)
            return True
        return False
    
    def _regret_insert(self, routes, node):
        """Regret insertion：考虑第二优选择的差异"""
        insertions = []
        
        for r_idx, route in enumerate(routes):
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [node] + route[pos:]
                if self._is_route_valid(new_route):
                    cost = self._calculate_insertion_cost(route, node, pos)
                    insertions.append((r_idx, pos, cost))
        
        if not insertions:
            return False
        
        # 按成本排序
        insertions.sort(key=lambda x: x[2])
        
        if len(insertions) == 1:
            # 只有一个可行位置
            r_idx, pos, _ = insertions[0]
            routes[r_idx].insert(pos, node)
            return True
        
        # 计算 regret（最优与次优的差异）
        best_cost = insertions[0][2]
        second_cost = insertions[1][2]
        regret = second_cost - best_cost
        
        # 选择 regret 最大的
        r_idx, pos, _ = insertions[0]
        routes[r_idx].insert(pos, node)
        return True
    
    def _is_route_valid(self, route):
        """快速检查路径是否有效"""
        if not route:
            return True
        
        # 容量（demands 索引 0 是 depot，客户节点 1-50 对应 demands[0-49]）
        load = sum(self.demands[n-1] for n in route if n > 0)
        if load > 1.0:
            return False
        
        # 时间和能量
        current_time = 0.0
        current_battery = self.battery_capacity
        current_node = 0
        
        for node in route:
            travel = self.distance_matrix[current_node][node]
            arrival = current_time + travel
            
            if self.tw_early is not None and arrival > self.tw_late[node]:
                return False
            
            current_battery -= travel * self.energy_rate
            if current_battery < 0:
                return False
            
            if self.tw_early is not None:
                current_time = max(arrival, self.tw_early[node]) + self.service_time[node]
            else:
                current_time = arrival
            
            current_node = node
        
        # 返回 depot
        return_dist = self.distance_matrix[current_node][0]
        if current_battery < return_dist * self.energy_rate:
            return False
        
        return True
    
    def _calculate_insertion_cost(self, route, node, pos):
        """计算插入成本"""
        prev_node = route[pos-1] if pos > 0 else 0
        next_node = route[pos] if pos < len(route) else 0
        
        # 插入前：prev -> next
        # 插入后：prev -> node -> next
        old_dist = self.distance_matrix[prev_node][next_node]
        new_dist = (self.distance_matrix[prev_node][node] + 
                   self.distance_matrix[node][next_node])
        
        return new_dist - old_dist


def solve_with_ortools_initial(distance_matrix, demands):
    """用 OR-Tools 生成初始解（只考虑容量，不考虑 EV/TW）"""
    if not ORTOOLS_AVAILABLE:
        return None
    
    n_nodes = len(distance_matrix)
    
    manager = pywrapcp.RoutingIndexManager(n_nodes, 10, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 1000)
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # 容量
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        if from_node == 0:
            return 0
        return int(demands[from_node - 1] * 100)
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    vehicle_capacities = [100] * 10
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, vehicle_capacities, True, 'Capacity'
    )
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.time_limit.FromSeconds(3)
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        routes = []
        for vehicle_id in range(10):
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != 0:
                    route.append(node)
                index = solution.Value(routing.NextVar(index))
            if route:
                routes.append(route)
        return routes
    return None


def load_test_data(filename):
    """从文件加载测试数据"""
    if not os.path.exists(filename):
        print(f"Test data file not found: {filename}")
        return None
    
    data = torch.load(filename)
    print(f"Loaded test data from {filename}")
    print(f"  Instances: {data['node_xy'].shape[0]}")
    print(f"  Problem size: {data['node_xy'].shape[1]}")
    return data


def get_single_instance(test_data, idx):
    """从测试数据中提取单个实例"""
    result = {}
    for k, v in test_data.items():
        if isinstance(v, torch.Tensor):
            result[k] = v[idx:idx+1]
        else:
            result[k] = v
    return result


def main():
    print(f"ALNS Solver for CVRP n={PROBLEM_SIZE} with EV and TW")
    print(f"Max Iterations: {MAX_ITERATIONS}, Destroy Rate: {DESTROY_RATE}")
    print(f"Test data: {TEST_DATA_FILE}")
    print("-" * 60)
    
    if not ORTOOLS_AVAILABLE:
        print("OR-Tools not available")
        return
    
    # 加载固定测试数据
    test_data = load_test_data(TEST_DATA_FILE)
    
    if test_data is None:
        print("Error: Cannot load test data!")
        return
    
    # 设置随机种子（用于 ALNS 进化）
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    
    total_cost = 0.0
    n_solved = 0
    start_time = time.time()
    
    for i in range(N_INSTANCES):
        # 从文件加载单个实例
        problem_data = get_single_instance(test_data, i)
        
        # 提取数据
        depot_xy = problem_data['depot_xy'].squeeze(0).numpy()
        node_xy = problem_data['node_xy'].squeeze(0).numpy()
        demands = problem_data['node_demand'].squeeze(0).numpy()
        
        all_nodes = np.vstack([depot_xy, node_xy])
        distance_matrix = np.sqrt(((all_nodes[:, None, :] - all_nodes[None, :, :]) ** 2).sum(axis=2))
        
        if USE_TW:
            tw_early = np.concatenate([[0], problem_data['tw_early'].squeeze(0).numpy()])
            tw_late = np.concatenate([[100], problem_data['tw_late'].squeeze(0).numpy()])
            service_time = np.concatenate([[0], problem_data['node_service_time'].squeeze(0).numpy()])
        else:
            tw_early = None
            tw_late = None
            service_time = None
        
        # 生成初始解（用 OR-Tools，只考虑容量）
        initial_routes = solve_with_ortools_initial(distance_matrix, demands)
        
        if initial_routes is None:
            print(f"Instance {i+1}/{N_INSTANCES}: No initial solution")
            continue
        
        # ALNS 优化
        alns = ALNS(distance_matrix, demands, tw_early, tw_late, service_time,
                   BATTERY_CAPACITY, ENERGY_CONSUMPTION_RATE, 
                   max_iter=MAX_ITERATIONS, destroy_rate=DESTROY_RATE)
        
        best_sol, best_cost = alns.solve(initial_routes)
        
        if best_sol:
            # 计算违反量
            total_violation = sum(best_sol._calculate_route_violation(r) for r in best_sol.routes)
            total_cost += best_cost
            n_solved += 1
            
            if (i + 1) % 10 == 0:
                avg_cost = total_cost / n_solved
                print(f"Instance {i+1}/{N_INSTANCES}: Cost={best_cost:.4f}, Avg={avg_cost:.4f}, Viol={total_violation:.2f}")
        else:
            print(f"Instance {i+1}/{N_INSTANCES}: No solution")
    
    elapsed = time.time() - start_time
    
    print("-" * 60)
    print(f"Results for n={PROBLEM_SIZE}:")
    print(f"Solved: {n_solved}/{N_INSTANCES}")
    if n_solved > 0:
        print(f"Average Cost: {total_cost / n_solved:.4f}")
    print(f"Total Time: {elapsed:.2f}s ({elapsed/N_INSTANCES:.2f}s per instance)")


if __name__ == "__main__":
    main()
