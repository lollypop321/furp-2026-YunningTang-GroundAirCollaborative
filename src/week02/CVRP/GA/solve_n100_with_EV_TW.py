##########################################################################################
# GA Solver for CVRP with EV and TW - n=100
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
from cvrp_env import CVRPGAEnv
from ga_solver import GeneticAlgorithm


##########################################################################################
# Parameters (与 POMO 保持一致)

PROBLEM_SIZE = 100
USE_EV = True
USE_TW = True

# EV 参数
EV_PARAMS = {
    'battery_capacity': 100.0,
    'energy_consumption_rate': 0.5,
}

# GA 参数（随问题规模增大）
POP_SIZE = 150
CX_PB = 0.85
MUT_PB = 0.02
N_GEN = 300

# 测试参数
N_INSTANCES = 100
TEST_DATA_FILE = '../vrp100_test_seed1234.pt'


def solve_single_instance(problem_data, instance_id):
    """求解单个实例"""
    env = CVRPGAEnv(problem_data, use_ev=USE_EV, use_tw=USE_TW)

    test_ind = list(range(1, PROBLEM_SIZE + 1))
    test_routes = env.decode_individual(test_ind)
    if len(test_routes) == 0 or sum(len(r) for r in test_routes) == 0:
        print(f"Warning: Instance {instance_id} has empty routes!")
        return float('inf'), 0, 0

    ga = GeneticAlgorithm(
        env=env,
        pop_size=POP_SIZE,
        cx_pb=CX_PB,
        mut_pb=MUT_PB,
        n_gen=N_GEN,
        use_elitism=True,
        elitism_rate=0.05
    )

    best_ind, best_cost = ga.run(verbose=False)

    if len(best_ind) != PROBLEM_SIZE or len(set(best_ind)) != PROBLEM_SIZE:
        print(f"  Warning: Instance {instance_id} invalid individual!")

    routes = env.decode_individual(best_ind)
    total_nodes = sum(len(r) for r in routes)
    if total_nodes != PROBLEM_SIZE:
        print(f"  Warning: Instance {instance_id} served {total_nodes}/{PROBLEM_SIZE} nodes!")
        best_cost = env.evaluate_individual(best_ind)

    tw_vio, energy_vio = env.count_violations(best_ind)
    return best_cost, tw_vio, energy_vio


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
    print(f"GA Solver for CVRP n={PROBLEM_SIZE} with EV and TW")
    print(f"Population: {POP_SIZE}, Generations: {N_GEN}")
    print(f"Test data: {TEST_DATA_FILE}")
    print("-" * 60)

    test_data = load_test_data(TEST_DATA_FILE)
    if test_data is None:
        return

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    total_cost = 0.0
    total_tw_vio = 0
    total_energy_vio = 0
    start_time = time.time()

    for i in range(N_INSTANCES):
        problem_data = get_single_instance(test_data, i)
        cost, tw_vio, energy_vio = solve_single_instance(problem_data, i)
        total_cost += cost
        total_tw_vio += tw_vio
        total_energy_vio += energy_vio

        if (i + 1) % 10 == 0:
            avg_cost = total_cost / (i + 1)
            print(f"Instance {i+1}/{N_INSTANCES}: Cost={cost:.4f}, Avg={avg_cost:.4f}")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Results for n={PROBLEM_SIZE}:")
    print(f"Average Cost: {total_cost / N_INSTANCES:.4f}")
    print(f"Average TW Violations: {total_tw_vio / N_INSTANCES:.4f}")
    print(f"Average Energy Violations: {total_energy_vio / N_INSTANCES:.4f}")
    print(f"Total Time: {elapsed:.2f}s ({elapsed/N_INSTANCES:.2f}s per instance)")


if __name__ == "__main__":
    main()
