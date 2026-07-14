import numpy as np
import pickle
import os

# ====================== 参数设置 ======================
CUSTOMER_NUM = 50
E_WINDOW = [0, 20]
L_WINDOW = [30, 80]
np.random.seed(50)  # 固定随机种子，确保可重复

# ====================== 算例生成函数 ======================
def build_instance(cust_n):
    depot = np.array([[0, 0]])
    custs = np.random.uniform(-5, 5, (cust_n, 2))
    nodes = np.vstack([depot, custs])
    n = len(nodes)
    dist = np.linalg.norm(nodes[:, None] - nodes[None, :], axis=-1)
    demand = np.random.randint(1, 10, size=cust_n)
    e = np.random.randint(E_WINDOW[0], E_WINDOW[1], cust_n)
    l = np.random.randint(L_WINDOW[0], L_WINDOW[1], cust_n)
    return nodes, dist, demand, e, l

# ====================== 生成并保存算例 ======================
if __name__ == "__main__":
    # 创建 instances 目录
    os.makedirs("instances", exist_ok=True)
    
    # 生成算例
    nodes, dist, demand, e, l = build_instance(CUSTOMER_NUM)
    
    # 保存到文件
    instance_data = {
        'nodes': nodes,
        'dist': dist,
        'demand': demand,
        'e': e,
        'l': l,
        'customer_num': CUSTOMER_NUM,
        'E_WINDOW': E_WINDOW,
        'L_WINDOW': L_WINDOW,
        'seed': 50
    }
    
    with open('instances/instance_n50_2.pkl', 'wb') as f:
        pickle.dump(instance_data, f)
    
    print("✅ 算例已生成并保存到 instances/instance_n50_2.pkl")
    print(f"   客户数量: {CUSTOMER_NUM}")
    print(f"   时间窗范围: E={E_WINDOW}, L={L_WINDOW}")
    print(f"   数据形状: nodes={nodes.shape}, dist={dist.shape}")