import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")

import torch
import time
from problem_def import get_random_problems_with_ev_tw
from cvrp_env import CVRPGAEnv

print("Step 1: Generating problem data...")
start = time.time()

problem_data = get_random_problems_with_ev_tw(
    batch_size=1,
    problem_size=50,
    use_ev=True,
    use_tw=True
)

print(f"  Done in {time.time()-start:.2f}s")
print(f"  Keys: {problem_data.keys()}")

print("\nStep 2: Creating environment...")
start = time.time()
env = CVRPGAEnv(problem_data, use_ev=True, use_tw=True)
print(f"  Done in {time.time()-start:.2f}s")
print(f"  Problem size: {env.problem_size}")

print("\nStep 3: Testing decode_individual...")
start = time.time()

test_ind = list(range(1, 51))
print(f"  Test individual: {test_ind[:10]}... (first 10)")

routes = env.decode_individual(test_ind)
print(f"  Done in {time.time()-start:.2f}s")
print(f"  Routes: {routes}")
print(f"  Total nodes served: {sum(len(r) for r in routes)}")

print("\nStep 4: Testing evaluate_individual...")
start = time.time()
cost = env.evaluate_individual(test_ind)
print(f"  Done in {time.time()-start:.2f}s")
print(f"  Cost: {cost}")

print("\nStep 5: Testing count_violations...")
start = time.time()
tw_vio, energy_vio = env.count_violations(test_ind)
print(f"  Done in {time.time()-start:.2f}s")
print(f"  TW violations: {tw_vio}, Energy violations: {energy_vio}")

print("\nAll tests passed!")

# 检查 distance_matrix 范围
print(f"\nDistance matrix stats:")
print(f"  Shape: {env.distance_matrix.shape}")
print(f"  Min: {env.distance_matrix.min():.4f}")
print(f"  Max: {env.distance_matrix.max():.4f}")
print(f"  Mean: {env.distance_matrix.mean():.4f}")

# 检查路由是否完整
print(f"\nRoute validation:")
all_nodes = set()
for r in routes:
    all_nodes.update(r)
print(f"  Nodes in routes: {len(all_nodes)} / 50")
print(f"  Missing nodes: {set(range(1, 51)) - all_nodes}")
