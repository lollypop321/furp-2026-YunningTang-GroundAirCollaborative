# generate_instance.py
# 运行此脚本一次，生成固定实例并保存到文件

import numpy as np
import pickle
import os
from datetime import datetime

# ====================== 参数设置（与主代码保持一致）======================
CUSTOMER_NUM = 50
E_WINDOW = [0, 5]
L_WINDOW = [15, 25]
np.random.seed(42)  # 固定种子确保可重复

def build_instance(cust_n):
    """生成算例"""
    depot = np.array([[0, 0]])
    custs = np.random.uniform(-10, 10, (cust_n, 2))
    nodes = np.vstack([depot, custs])
    dist = np.linalg.norm(nodes[:, None] - nodes[None, :], axis=-1)
    demand = np.random.randint(1, 10, size=cust_n)
    e = np.random.randint(E_WINDOW[0], E_WINDOW[1], cust_n)
    l = np.random.randint(L_WINDOW[0], L_WINDOW[1], cust_n)
    return nodes, dist, demand, e, l

# ====================== 生成实例 ======================
print("="*60)
print("生成固定算例...")
print("="*60)

nodes, dist_mat, demand_list, e_list, l_list = build_instance(CUSTOMER_NUM)

# ====================== 保存到文件（二进制格式）======================
instance_data = {
    'nodes': nodes,
    'dist_mat': dist_mat,
    'demand_list': demand_list,
    'e_list': e_list,
    'l_list': l_list,
    'CUSTOMER_NUM': CUSTOMER_NUM,
    'E_WINDOW': E_WINDOW,
    'L_WINDOW': L_WINDOW,
    'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'seed': 42
}

with open('fixed_instance.pkl', 'wb') as f:
    pickle.dump(instance_data, f)

print("✅ 实例已保存到: fixed_instance.pkl")
print(f"\n📊 实例信息:")
print(f"  客户数量: {CUSTOMER_NUM}")
print(f"  坐标范围: {nodes.min():.2f} ~ {nodes.max():.2f}")
print(f"  时间窗范围: e=[{e_list.min()}, {e_list.max()}], l=[{l_list.min()}, {l_list.max()}]")
print(f"  需求范围: {demand_list.min()} ~ {demand_list.max()}")
print(f"  生成时间: {instance_data['generated_at']}")

# ====================== 可选：保存为可读文本格式 ======================
with open('instance_info.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("固定算例详细信息\n")
    f.write("="*60 + "\n\n")
    
    f.write("【客户坐标】\n")
    f.write("客户编号: (x, y)\n")
    for i in range(1, len(nodes)):
        f.write(f"  {i:3d}: ({nodes[i,0]:8.4f}, {nodes[i,1]:8.4f})\n")
    
    f.write("\n【时间窗】\n")
    f.write("客户编号: [最早到达, 最晚到达]\n")
    for i in range(CUSTOMER_NUM):
        f.write(f"  {i+1:3d}: [{e_list[i]:3d}, {l_list[i]:3d}]\n")
    
    f.write("\n【需求】\n")
    f.write("客户编号: 需求量\n")
    for i in range(CUSTOMER_NUM):
        f.write(f"  {i+1:3d}: {demand_list[i]:3d}\n")

print("✅ 可读文本已保存到: instance_info.txt")
print("\n" + "="*60)