import torch
import random
import sys
import os

# Path Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

from CVRProblemDef import get_random_problems_with_ev_tw

# 设置随机种子，确保可复现
random.seed(1234)
torch.manual_seed(1234)

# 生成 100 个测试实例（与 GA/OR/ALNS 一致）
print("Generating 100 test instances for CVRP50 with EV and TW constraints...")
test_data = get_random_problems_with_ev_tw(
    batch_size=100,
    problem_size=50,
    use_ev=True,
    use_tw=True
)

# 添加缺失的字段（与 GA/problem_def.py 保持一致）
test_data['battery_capacity'] = 100.0
test_data['energy_consumption_rate'] = 0.5

# 保存测试集
output_path = '../vrp50_test_seed1234.pt'
torch.save(test_data, output_path)
print(f"Test dataset saved to: {output_path}")
print(f"Keys: {test_data.keys()}")
print(f"Depot xy shape: {test_data['depot_xy'].shape}")
print(f"Node xy shape: {test_data['node_xy'].shape}")
print(f"Node demand shape: {test_data['node_demand'].shape}")
print(f"Node service time shape: {test_data['node_service_time'].shape}")
print(f"TW early shape: {test_data['tw_early'].shape}")
print(f"TW late shape: {test_data['tw_late'].shape}")
print(f"Initial battery shape: {test_data['initial_battery'].shape}")
