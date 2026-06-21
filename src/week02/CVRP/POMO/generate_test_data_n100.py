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

# 生成 n=100 测试实例
print("Generating 100 test instances for CVRP100 with EV and TW constraints...")
test_data = get_random_problems_with_ev_tw(
    batch_size=100,
    problem_size=100,
    use_ev=True,
    use_tw=True
)

# 添加缺失的字段
test_data['battery_capacity'] = 100.0
test_data['energy_consumption_rate'] = 0.5

# 保存测试集
output_path = '../vrp100_test_seed1234.pt'
torch.save(test_data, output_path)
print(f"Test dataset saved to: {output_path}")
print(f"Keys: {test_data.keys()}")
print(f"Depot xy shape: {test_data['depot_xy'].shape}")
print(f"Node xy shape: {test_data['node_xy'].shape}")
