##########################################################################################
# OR-Tools Solver for CVRP with EV and TW
# 使用 Google OR-Tools 求解，与 POMO 和 GA 保持一致的问题设置
##########################################################################################

import os
import sys
import torch
import numpy as np
import random
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")

from problem_def import get_random_problems_with_ev_tw

# 尝试导入 OR-Tools
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("Warning: OR-Tools not installed. Install with: pip install ortools")


##########################################################################################
# Parameters (与 POMO 保持一致)

PROBLEM_SIZE = 50
USE_EV = False  # OR-Tools 暂不处理能量约束
USE_TW = True   # 启用时间窗约束

# EV 参数
BATTERY_CAPACITY = 100.0
ENERGY_CONSUMPTION_RATE = 0.5

# 测试参数
N_INSTANCES = 100
TEST_DATA_FILE = '../vrp50_test_seed1234.pt'  # 固定测试数据文件


def solve_with_ortools(distance_matrix, demands, tw_early, tw_late, service_time,
                       vehicle_capacity, battery_capacity, energy_rate, use_ev=True, use_tw=True):
    """
    使用 OR-Tools 求解 CVRP-EV-TW
    """
    if not ORTOOLS_AVAILABLE:
        return None, 0, 0

    n_nodes = len(distance_matrix)
    n_customers = n_nodes - 1  # 去掉 depot

    # 创建路由模型
    # 估计需要的车辆数：总需求 / 车辆容量 + 缓冲
    total_demand = sum(demands)
    n_vehicles = max(1, int(total_demand / vehicle_capacity) + 5)
    n_vehicles = min(n_vehicles, 20)  # 最多 20 辆车
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, 0)  # n_vehicles 辆车，depot=0
    routing = pywrapcp.RoutingModel(manager)

    # 距离回调函数
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # OR-Tools 使用整数，将距离放大 1000 倍
        return int(distance_matrix[from_node][to_node] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 添加容量约束
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        # depot 的需求为 0
        if from_node == 0:
            return 0
        return int(demands[from_node - 1] * 100)  # 客户节点索引-1

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    vehicle_capacities = [int(vehicle_capacity * 100)] * n_vehicles
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        vehicle_capacities,  # 每辆车的容量
        True,  # fix start cumul to zero
        'Capacity'
    )

    # 添加时间窗约束（如果启用）
    if use_tw:
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # 行驶时间 + 服务时间（depot 服务时间为 0）
            travel_time = distance_matrix[from_node][to_node]
            if to_node == 0:
                service = 0
            else:
                service = service_time[to_node]
            return int((travel_time + service) * 1000)

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            int(1000 * 100),  # 允许等待时间（大幅放宽）
            int(1000 * 100),  # 最大时间（大幅放宽）
            False,  # 不强制从零开始
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')

        # 为每个客户添加时间窗（depot 的时间窗设为 [0, 100]）
        for customer_id in range(n_nodes):
            index = manager.NodeToIndex(customer_id)
            if customer_id == 0:
                # depot 时间窗宽松
                time_dimension.CumulVar(index).SetRange(0, int(100 * 1000))
            else:
                time_dimension.CumulVar(index).SetRange(
                    int(tw_early[customer_id] * 1000),
                    int(tw_late[customer_id] * 1000)
                )

    # 搜索参数
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(5)  # 最多 5 秒

    # 求解
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        # 计算总距离
        total_distance = solution.ObjectiveValue() / 1000.0
        return total_distance, 0, 0  # OR-Tools 保证约束满足
    else:
        return None, 0, 0


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
    print(f"OR-Tools Solver for CVRP n={PROBLEM_SIZE} with EV and TW")
    print(f"Test data: {TEST_DATA_FILE}")
    print("-" * 60)

    if not ORTOOLS_AVAILABLE:
        print("OR-Tools not available. Please install: pip install ortools")
        return

    # 加载固定测试数据
    test_data = load_test_data(TEST_DATA_FILE)
    
    if test_data is None:
        print("Error: Cannot load test data!")
        return

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

        # 构建距离矩阵
        all_nodes = np.vstack([depot_xy, node_xy])
        distance_matrix = np.sqrt(((all_nodes[:, None, :] - all_nodes[None, :, :]) ** 2).sum(axis=2))

        # 提取时间窗
        if USE_TW:
            tw_early_np = problem_data['tw_early'].squeeze(0).numpy()
            tw_late_np = problem_data['tw_late'].squeeze(0).numpy()
            service_time_np = problem_data['node_service_time'].squeeze(0).numpy()
            # depot 索引为 0，客户索引为 1-50
            tw_early = np.concatenate([[0], tw_early_np])
            tw_late = np.concatenate([[100], tw_late_np])
            service_time = np.concatenate([[0], service_time_np])
            

        else:
            tw_early = None
            tw_late = None
            service_time = None

        # 求解
        cost, tw_vio, energy_vio = solve_with_ortools(
            distance_matrix, demands, tw_early, tw_late, service_time,
            vehicle_capacity=1.0,
            battery_capacity=BATTERY_CAPACITY,
            energy_rate=ENERGY_CONSUMPTION_RATE,
            use_ev=USE_EV,
            use_tw=USE_TW
        )

        if cost is not None:
            total_cost += cost
            n_solved += 1

            if (i + 1) % 10 == 0:
                avg_cost = total_cost / n_solved
                print(f"Instance {i+1}/{N_INSTANCES}: Cost={cost:.4f}, Avg={avg_cost:.4f}")
        else:
            print(f"Instance {i+1}/{N_INSTANCES}: No solution found")

    elapsed = time.time() - start_time

    print("-" * 60)
    print(f"Results for n={PROBLEM_SIZE}:")
    print(f"Solved: {n_solved}/{N_INSTANCES}")
    if n_solved > 0:
        print(f"Average Cost: {total_cost / n_solved:.4f}")
    print(f"Total Time: {elapsed:.2f}s ({elapsed/N_INSTANCES:.2f}s per instance)")


if __name__ == "__main__":
    main()
