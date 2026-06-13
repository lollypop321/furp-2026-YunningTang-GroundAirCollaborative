# Simple Smoke Test for OR-Tools & VRP Environment
import pandas as pd
import ortools
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Check imports and versions
print("All libraries imported successfully!")
print("pandas version:", pd.__version__)
print("ortools version:", ortools.__version__)


def create_data_model():
    """Small VRP instance for smoke test."""
    data = {}
    data["distance_matrix"] = [
        [0, 10, 20],
        [10, 0, 15],
        [20, 15, 0]
    ]
    data["num_vehicles"] = 1
    data["depot"] = 0
    return data


def print_solution(data, manager, routing, solution):
    """Print route and distance details."""
    print(f"\nSolution found! Objective: {solution.ObjectiveValue()}")
    max_route_distance = 0

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0

        while not routing.IsEnd(index):
            plan_output += f" {manager.IndexToNode(index)} ->"
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f" {manager.IndexToNode(index)}\n"
        plan_output += f"Distance: {route_distance}m\n"
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)

    print(f"Max route distance: {max_route_distance}m")


def main():
    data = create_data_model()

    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]),
        data["num_vehicles"],
        data["depot"]
    )

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return data["distance_matrix"][from_node][to_node]

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_params)

    if solution:
        print_solution(data, manager, routing, solution)
    else:
        print("\nNo solution found.")


if __name__ == "__main__":
    main()