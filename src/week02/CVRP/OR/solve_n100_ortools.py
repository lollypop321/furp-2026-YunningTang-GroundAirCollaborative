##########################################################################################
# OR-Tools Solver for CVRP with EV and TW - n=100
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

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


##########################################################################################
# Parameters

PROBLEM_SIZE = 100
USE_EV = False
USE_TW = True

BATTERY_CAPACITY = 100.0
ENERGY_CONSUMPTION_RATE = 0.5

N_INSTANCES = 100
TEST_DATA_FILE = '../vrp100_test_seed1234.pt'


def solve_with_ortools(distance_matrix, demands, tw_early, tw_late, service_time,
                       vehicle_capacity, battery_capacity, energy_rate, use_ev=True, use_tw=True):
    if not ORTOOLS_AVAILABLE:
        return None, 0, 0

    n_nodes = len(distance_matrix)
    total_demand = sum(demands)
    n_vehicles = max(1, int(total_demand / vehicle_capacity) + 5)
    n_vehicles = min(n_vehicles, 30)
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(demands[from_node] * 100)

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, [int(vehicle_capacity * 100)] * n_vehicles, True, 'Capacity')

    if use_tw and tw_early is not None and tw_late is not None:
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            travel_time = distance_matrix[from_node][to_node]
            service = service_time[from_node] if from_node < len(service_time) else 0
            return int((travel_time + service) * 1000)

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(time_callback_index, 0, int(15 * 1000), False, 'Time')
        time_dimension = routing.GetDimensionOrDie('Time')

        for node in range(1, n_nodes):
            index = manager.NodeToIndex(node)
            time_dimension.CumulVar(index).SetRange(int(tw_early[node] * 1000), int(tw_late[node] * 1000))

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(10)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        total_distance = solution.ObjectiveValue() / 1000.0
        return total_distance, 0, 0
    else:
        return None, 0, 0


def load_test_data(filename):
    if not os.path.exists(filename):
        print(f"Test data file not found: {filename}")
        return None
    data = torch.load(filename)
    print(f"Loaded test data from {filename}")
    print(f"  Instances: {data['node_xy'].shape[0]}")
    print(f"  Problem size: {data['node_xy'].shape[1]}")
    return data


def get_single_instance(test_data, idx):
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
        print("OR-Tools not available")
        return

    test_data = load_test_data(TEST_DATA_FILE)
    if test_data is None:
        return

    total_cost = 0.0
    n_solved = 0
    start_time = time.time()

    for i in range(N_INSTANCES):
        problem_data = get_single_instance(test_data, i)

        depot_xy = problem_data['depot_xy'].squeeze(0).numpy()
        node_xy = problem_data['node_xy'].squeeze(0).numpy()
        demands = problem_data['node_demand'].squeeze(0).numpy()

        n = PROBLEM_SIZE
        coords = np.vstack([depot_xy, node_xy])
        distance_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))

        demands = np.concatenate([[0], demands])
        vehicle_capacity = 1.0

        if USE_TW and 'tw_early' in problem_data:
            tw_early = np.concatenate([[0], problem_data['tw_early'].squeeze(0).numpy()])
            tw_late = np.concatenate([[100], problem_data['tw_late'].squeeze(0).numpy()])
            service_time = np.concatenate([[0], problem_data['node_service_time'].squeeze(0).numpy()])
        else:
            tw_early = tw_late = service_time = None

        cost, tw_vio, energy_vio = solve_with_ortools(
            distance_matrix, demands, tw_early, tw_late, service_time,
            vehicle_capacity, BATTERY_CAPACITY, ENERGY_CONSUMPTION_RATE, USE_EV, USE_TW
        )

        if cost is not None:
            total_cost += cost
            n_solved += 1

        if (i + 1) % 10 == 0:
            avg_cost = total_cost / n_solved if n_solved > 0 else 0
            cost_str = f"{cost:.4f}" if cost else "N/A"
            print(f"Instance {i+1}/{N_INSTANCES}: Cost={cost_str}, Avg={avg_cost:.4f}")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Results for n={PROBLEM_SIZE}:")
    print(f"Solved: {n_solved}/{N_INSTANCES}")
    if n_solved > 0:
        print(f"Average Cost: {total_cost / n_solved:.4f}")
    print(f"Total Time: {elapsed:.2f}s ({elapsed/N_INSTANCES:.2f}s per instance)")


if __name__ == "__main__":
    main()
