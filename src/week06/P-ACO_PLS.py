import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from math import inf
from datetime import datetime
import time

# ====================== 参数配置 ======================
TRUCK_SPEED = 1.0
DRONE_SPEED = 2.0
ENDURANCE = 4
DRONE_MAX_RANGE = ENDURANCE
DEPOT = 0

# 50客户
TRUCK_NUM = 4
DRONE_NUM = TRUCK_NUM
CUSTOMER_NUM = 50
K = 70
MAX_IT = 170

# # 100客户
# TRUCK_NUM = 8
# DRONE_NUM = TRUCK_NUM
# CUSTOMER_NUM = 100
# K = 120
# MAX_IT = 250

ALPHA = 1
BETA = 2
q0 = 0.5
RHO = 0.15
xi = 0.1
Qt = 60
Qc = 120
C = 1000
tau0 = 10
tau_max = 20
tau_min = 1
MAX_EAN = 80
PARALLEL_GROUP = 4

# # 50客户时间窗
# E_WINDOW = [0, 5]
# L_WINDOW = [18, 28]

E_WINDOW = [0, 3]
L_WINDOW = [12, 18]

# 每辆卡车最多服务的无人机客户数
max_drone_per_truck = 4

# 无人机操作参数
s_l = 0.2
s_r = 0.2
M = 10000

# ===== PLS参数 =====
PLS_INTERVAL = 5      # 每5代触发一次PLS
PLS_MAX_ITER = 2      # 每个解的搜索轮数

np.random.seed(42)
random.seed(42)

# ====================== 算例生成 ======================
def build_instance(cust_n):
    depot = np.array([[0, 0]])
    if cust_n <= 25:
        scale = 5
    elif cust_n <= 50:
        scale = 10
    else:
        scale = 15
    custs = np.random.uniform(-scale, scale, (cust_n, 2))
    nodes = np.vstack([depot, custs])
    n = len(nodes)
    dist = np.linalg.norm(nodes[:, None] - nodes[None, :], axis=-1)
    demand = np.random.randint(1, 10, size=cust_n)
    e = np.random.randint(E_WINDOW[0], E_WINDOW[1], cust_n)
    l = np.random.randint(L_WINDOW[0], L_WINDOW[1], cust_n)
    return nodes, dist, demand, e, l

nodes, dist_mat, demand_list, e_list, l_list = build_instance(CUSTOMER_NUM)
N = len(nodes)
CUST_START = 1
CUST_END = CUSTOMER_NUM

print("="*60)
print("P-ACO + PLS 混合优化 (HMOA风格)")
print("="*60)
print(f"客户数量: {CUSTOMER_NUM}")
print(f"卡车数量: {TRUCK_NUM}")
print(f"无人机数量: {DRONE_NUM}")
print(f"无人机续航: {ENDURANCE}")
print(f"最大迭代: {MAX_IT}")
print(f"蚂蚁数量: {K}")
print(f"蚂蚁性格: 5种 (delay_first/cost_first/balance/delay_favor/cost_favor)")
print(f"PLS触发间隔: 每{PLS_INTERVAL}代")
print("="*60 + "\n")

# ====================== 帕累托支配 ======================
def dominates(obj1, obj2):
    c1, t1 = obj1
    c2, t2 = obj2
    return (c1 <= c2 and t1 <= t2) and (c1 < c2 or t1 < t2)

def crowding_distance(pareto_objs):
    n = len(pareto_objs)
    if n <= 2:
        return np.full(n, inf, dtype=np.float64)
    dist = np.zeros(n, dtype=np.float64)
    cost = np.array([p[0] for p in pareto_objs], dtype=np.float64)
    tard = np.array([p[1] for p in pareto_objs], dtype=np.float64)
    idx_cost = np.argsort(cost)
    idx_tard = np.argsort(tard)
    dist[idx_cost[0]] = inf
    dist[idx_cost[-1]] = inf
    cost_range = cost.max() - cost.min()
    tard_range = tard.max() - tard.min()
    if cost_range > 1e-9:
        for i in range(1, n-1):
            prev = idx_cost[i-1]
            nxt = idx_cost[i+1]
            dist[idx_cost[i]] += abs(cost[nxt] - cost[prev]) / cost_range
    if tard_range > 1e-9:
        for i in range(1, n-1):
            prev = idx_tard[i-1]
            nxt = idx_tard[i+1]
            dist[idx_tard[i]] += abs(tard[nxt] - tard[prev]) / tard_range
    return dist

def update_archive(archive, new_sol, similarity_threshold=0.5):
    """更新帕累托档案，保护边界解"""
    if not archive:
        return [new_sol]
    
    # ===== 第一步：检查是否与现有解重复 =====
    for sol in archive:
        dist = np.sqrt((new_sol[0] - sol[0])**2 + (new_sol[1] - sol[1])**2)
        if dist < similarity_threshold:
            return archive  # 重复解，直接返回
    
    # ===== 第二步：正常的支配检查 =====
    dominated_by_existing = False
    to_remove = []
    
    for i, sol in enumerate(archive):
        if dominates(sol, new_sol):
            dominated_by_existing = True
            break
        if dominates(new_sol, sol):
            to_remove.append(i)
    
    if dominated_by_existing:
        return archive
    
    new_arch = []
    for i, sol in enumerate(archive):
        if i not in to_remove:
            new_arch.append(sol)
    
    new_arch.append(new_sol)
    
    # ===== 第三步：裁剪时保护边界解 =====
    if len(new_arch) > MAX_EAN:
        cd = crowding_distance(new_arch)
        cost_min_idx = np.argmin([s[0] for s in new_arch])
        delay_min_idx = np.argmin([s[1] for s in new_arch])
        cd[cost_min_idx] = inf
        cd[delay_min_idx] = inf
        min_idx = np.argmin(cd)
        new_arch.pop(min_idx)
    
    return new_arch

# ====================== 评估函数 ======================
def evaluate_solution(truck_routes, drone_assigns, unvisited):
    if len(unvisited) > 0:
        return inf, inf, [], [], False
    total_truck_cost = 0.0
    total_drone_cost = 0.0
    total_tard = 0.0
    truck_edges = []
    drone_edges = []

    all_truck_customers = set()
    for route in truck_routes:
        for node in route:
            if node != DEPOT:
                all_truck_customers.add(node)

    all_drone_customers = set()
    for dm in drone_assigns:
        for key in dm:
            if key != '_intervals':
                all_drone_customers.add(key)

    overlap = all_truck_customers & all_drone_customers
    if overlap:
        return inf, inf, [], [], False

    for tid, route in enumerate(truck_routes):
        route_set = set(route)
        drone_map = drone_assigns[tid]
        for cust, val in drone_map.items():
            if cust == '_intervals':
                continue
            launch, land = val
            if launch not in route_set or land not in route_set:
                return inf, inf, [], [], False

    for tid, route in enumerate(truck_routes):
        drone_map = drone_assigns[tid]
        truck_time = {int(DEPOT): 0.0}

        for k in range(len(route)-1):
            i, j = int(route[k]), int(route[k+1])
            d = dist_mat[i, j]
            total_truck_cost += d
            t = d / TRUCK_SPEED
            truck_time[j] = truck_time[i] + t
            truck_edges.append((i, j, tid))

        for node in route:
            node = int(node)
            if node == DEPOT or node in all_drone_customers:
                continue
            st = truck_time[node]
            early = max(0.0, e_list[node-1] - st)
            late = max(0.0, st - l_list[node-1])
            total_tard += early + late

        drone_tasks = [(cust, launch, land) for cust, (launch, land) in drone_map.items() if cust != '_intervals']
        drone_tasks.sort(key=lambda x: truck_time[x[1]])
        
        prev_return = 0.0
        for cust, launch, land in drone_tasks:
            d1 = dist_mat[launch, cust]
            d2 = dist_mat[cust, land]
            total = d1 + d2
            if total > DRONE_MAX_RANGE + 1e-6:
                return inf, inf, [], [], False
            total_drone_cost += total
            drone_edges.append((launch, cust, tid))
            drone_edges.append((cust, land, tid))
            
            launch_time = truck_time[launch]
            if launch_time < prev_return:
                return inf, inf, [], [], False
            
            arrival_at_cust = launch_time + s_l + d1 / DRONE_SPEED
            return_time = launch_time + s_l + total / DRONE_SPEED + s_r
            
            if return_time > truck_time[land] + 1e-6:
                return inf, inf, [], [], False
            
            early = max(0.0, e_list[cust-1] - arrival_at_cust)
            late = max(0.0, arrival_at_cust - l_list[cust-1])
            total_tard += early + late
            prev_return = return_time

    total_cost = total_truck_cost + total_drone_cost
    return total_cost, total_tard, truck_edges, drone_edges, True


# ====================== 蚂蚁构造（5种性格） ======================
def ant_construct(tau_truck, tau_tard_truck, tau_drone, tau_tard_drone, mode=None):
    if mode is None:
        mode = random.choice(['delay_first', 'cost_first', 'balance', 'delay_favor', 'cost_favor'])
    
    if mode == 'delay_first':
        delay_weight = 15.0
        cost_weight = 1.0
    elif mode == 'cost_first':
        delay_weight = 1.0
        cost_weight = 15.0
    elif mode == 'balance':
        delay_weight = 5.0
        cost_weight = 5.0
    elif mode == 'delay_favor':
        delay_weight = 8.0
        cost_weight = 3.0
    else:
        delay_weight = 3.0
        cost_weight = 8.0
    
    # unvisited = set(range(CUST_START, CUST_END+1))

    n_customers = len(nodes) - 1  # 减去仓库节点 (DEPOT)
    unvisited = set(range(1, n_customers + 1))  # 客户编号从 1 到 n_customers
    
    truck_routes = [[DEPOT] for _ in range(TRUCK_NUM)]
    truck_time = {i: 0.0 for i in range(TRUCK_NUM)}
    drone_assigns = [dict() for _ in range(TRUCK_NUM)]

    while unvisited:
        valid_trucks = [i for i in range(TRUCK_NUM) if truck_routes[i]]
        if not valid_trucks:
            break
        min_tid = min(valid_trucks, key=lambda i: truck_time[i])
        
        cur_node = truck_routes[min_tid][-1]
        candidates = list(unvisited)
        
        if not candidates:
            break
            
        r = random.random()
        r_weight = random.uniform(0, 1)

        if r <= q0:
            max_p = -1
            sel_j = candidates[0]
            for j in candidates:
                tau = r_weight * tau_truck[cur_node, j] + (1 - r_weight) * tau_tard_truck[cur_node, j]
                eta_cost = C / (dist_mat[cur_node, j] + 1e-6)
                pred_t = truck_time[min_tid] + dist_mat[cur_node, j] / TRUCK_SPEED
                penalty = max(0, e_list[j-1] - pred_t) + max(0, pred_t - l_list[j-1]) + 1e-6
                eta_tard = C / penalty
                prob = (tau**ALPHA) * (eta_cost * eta_tard)**BETA
                if prob > max_p:
                    max_p = prob
                    sel_j = j
        else:
            weights = []
            for j in candidates:
                tau = r_weight * tau_truck[cur_node, j] + (1 - r_weight) * tau_tard_truck[cur_node, j]
                eta_cost = C / (dist_mat[cur_node, j] + 1e-6)
                pred_t = truck_time[min_tid] + dist_mat[cur_node, j] / TRUCK_SPEED
                penalty = max(0, e_list[j-1] - pred_t) + max(0, pred_t - l_list[j-1]) + 1e-6
                eta_tard = C / penalty
                w = (tau**ALPHA) * (eta_cost * eta_tard)**BETA
                weights.append(w)
            weights = np.array(weights) / (sum(weights) + 1e-6)
            sel_j = np.random.choice(candidates, p=weights)

        sel_j_int = int(sel_j)
        truck_routes[min_tid].append(sel_j_int)
        unvisited.remove(sel_j_int)
        add_t = dist_mat[cur_node, sel_j_int] / TRUCK_SPEED
        truck_time[min_tid] += add_t

        tau_truck[cur_node, sel_j_int] = (1 - xi) * tau_truck[cur_node, sel_j_int] + xi * tau0
        tau_tard_truck[cur_node, sel_j_int] = (1 - xi) * tau_tard_truck[cur_node, sel_j_int] + xi * tau0

    for tid in range(TRUCK_NUM):
        if truck_routes[tid] and truck_routes[tid][-1] != DEPOT:
            last = truck_routes[tid][-1]
            back_t = dist_mat[last, DEPOT] / TRUCK_SPEED
            truck_time[tid] += back_t
            truck_routes[tid].append(DEPOT)
        elif truck_routes[tid]:
            pass
        else:
            truck_routes[tid] = [DEPOT, DEPOT]

    # ===== 无人机分配 =====
    global_drone_customers = set()
    original_routes = [route.copy() for route in truck_routes]

    for tid in range(TRUCK_NUM):
        raw_route = original_routes[tid].copy()
        drone_map = drone_assigns[tid]
        
        if len(raw_route) < 3:
            continue
        
        raw_truck_time = {DEPOT: 0.0}
        for k in range(len(raw_route)-1):
            a = raw_route[k]
            b = raw_route[k+1]
            raw_truck_time[b] = raw_truck_time[a] + dist_mat[a, b] / TRUCK_SPEED

        candidates = [node for node in raw_route if node != DEPOT and node not in global_drone_customers]
        candidates.sort(key=lambda x: l_list[x-1] - e_list[x-1])

        assigned = 0

        for cust in candidates:
            if assigned >= max_drone_per_truck:
                break
            
            try:
                cust_idx = raw_route.index(cust)
            except ValueError:
                continue
            
            best_pair = None
            best_score = -inf
            
            truck_arrival_time = 0
            for k in range(cust_idx):
                truck_arrival_time += dist_mat[raw_route[k], raw_route[k+1]] / TRUCK_SPEED
            
            truck_late = max(0, truck_arrival_time - l_list[cust-1])
            
            for i_idx in range(cust_idx):
                i = raw_route[i_idx]
                if i == DEPOT:
                    continue
                for k_idx in range(cust_idx + 1, len(raw_route)):
                    k = raw_route[k_idx]
                    if k == DEPOT:
                        continue

                    d_total = dist_mat[i, cust] + dist_mat[cust, k]
                    if d_total > DRONE_MAX_RANGE + 1e-6:
                        continue

                    t_fly = d_total / DRONE_SPEED
                    t_back = raw_truck_time[i] + s_l + t_fly + s_r
                    if t_back > raw_truck_time[k] + 0.1:
                        continue

                    conflict = False
                    new_s = raw_truck_time[i]
                    new_e = raw_truck_time[k]
                    for (s0, e0) in drone_map.get('_intervals', []):
                        if not (new_e <= s0 or new_s >= e0):
                            conflict = True
                            break
                    if conflict:
                        continue

                    drone_arrival = raw_truck_time[i] + s_l + dist_mat[i, cust] / DRONE_SPEED
                    drone_late = max(0, drone_arrival - l_list[cust-1])
                    tard_improvement = truck_late - drone_late
                    
                    truck_dist = dist_mat[i, cust] + dist_mat[cust, k]
                    savings = truck_dist - d_total
                    
                    score = tard_improvement * delay_weight + savings * cost_weight
                    
                    if truck_late > 0 and drone_late == 0:
                        score += 50.0 * (delay_weight / 10.0)
                    
                    if score > best_score:
                        best_score = score
                        best_pair = (i, k, cust_idx)

            if best_pair:
                i, k, cust_idx = best_pair
                drone_map[cust] = (i, k)
                global_drone_customers.add(cust)
                assigned += 1
                
                if '_intervals' not in drone_map:
                    drone_map['_intervals'] = []
                drone_map['_intervals'].append((raw_truck_time[i], raw_truck_time[k]))

        new_route = []
        for node in raw_route:
            if node not in global_drone_customers:
                new_route.append(int(node))
        if not new_route:
            new_route = [DEPOT, DEPOT]
        else:
            if new_route[0] != DEPOT:
                new_route.insert(0, DEPOT)
            if new_route[-1] != DEPOT:
                new_route.append(DEPOT)
        truck_routes[tid] = new_route

    for dm in drone_assigns:
        dm.pop('_intervals', None)

    cost, tard, te, de, feasible = evaluate_solution(truck_routes, drone_assigns, set())
    return cost, tard, truck_routes, drone_assigns, te, de, feasible, mode


# ====================== HMOA邻域算子 ======================

def operator_truck_to_drone(truck_routes, drone_assigns, tid):
    """
    N1: 将卡车路径中最昂贵的客户改为无人机服务
    """
    route = truck_routes[tid]
    if len(route) <= 3:
        return None
    
    # 找卡车路径中最昂贵的无人机可服务节点
    best_cust = None
    best_saving = -inf
    
    for idx in range(1, len(route)-1):
        cust = route[idx]
        if cust == DEPOT:
            continue
        already_drone = False
        for dm in drone_assigns:
            if cust in dm:
                already_drone = True
                break
        if already_drone:
            continue
        
        prev_node = route[idx-1]
        next_node = route[idx+1]
        saving = dist_mat[prev_node, cust] + dist_mat[cust, next_node] - dist_mat[prev_node, next_node]
        
        if saving > best_saving:
            best_saving = saving
            best_cust = (idx, cust)
    
    if best_cust is None or best_saving <= 0:
        return None
    
    idx, cust = best_cust
    
    # 从卡车路径中移除
    new_route = route[:idx] + route[idx+1:]
    
    # 尝试用无人机服务该客户
    # 计算卡车时间
    truck_time = {DEPOT: 0.0}
    for k in range(len(new_route)-1):
        i, j = new_route[k], new_route[k+1]
        truck_time[j] = truck_time[i] + dist_mat[i, j] / TRUCK_SPEED
    
    best_launch = None
    best_land = None
    best_drone_cost = inf
    
    for i_idx in range(len(new_route)-1):
        launch = new_route[i_idx]
        if launch == DEPOT:
            continue
        for k_idx in range(i_idx+1, len(new_route)):
            land = new_route[k_idx]
            if land == DEPOT:
                continue
            
            d_total = dist_mat[launch, cust] + dist_mat[cust, land]
            if d_total > DRONE_MAX_RANGE:
                continue
            
            launch_time = truck_time.get(launch, 0)
            drone_arrival = launch_time + s_l + dist_mat[launch, cust] / DRONE_SPEED
            return_time = launch_time + s_l + d_total / DRONE_SPEED + s_r
            
            if return_time > truck_time.get(land, inf) + 1e-6:
                continue
            
            if d_total < best_drone_cost:
                best_drone_cost = d_total
                best_launch = launch
                best_land = land
    
    if best_launch is None:
        return None
    
    # 检查该客户是否已被其他卡车的无人机服务
    for dm in drone_assigns:
        if cust in dm:
            return None
    
    new_truck_routes = truck_routes.copy()
    new_truck_routes[tid] = new_route
    
    new_drone_assigns = [d.copy() for d in drone_assigns]
    new_drone_assigns[tid][cust] = (best_launch, best_land)
    
    return new_truck_routes, new_drone_assigns


def operator_drone_to_truck(truck_routes, drone_assigns, tid):
    """
    N2: 将无人机任务中最昂贵的客户移回卡车
    """
    drone_map = drone_assigns[tid]
    if not drone_map:
        return None
    
    most_expensive = None
    max_cost = -inf
    
    for cust, (launch, land) in drone_map.items():
        if cust == '_intervals':
            continue
        cost = dist_mat[launch, cust] + dist_mat[cust, land]
        if cost > max_cost:
            max_cost = cost
            most_expensive = cust
    
    if most_expensive is None:
        return None
    
    route = truck_routes[tid]
    best_pos = 1
    min_insert_cost = inf
    
    for pos in range(1, len(route)):
        prev_node = route[pos-1]
        next_node = route[pos] if pos < len(route) else DEPOT
        insert_cost = dist_mat[prev_node, most_expensive] + dist_mat[most_expensive, next_node] - dist_mat[prev_node, next_node]
        if insert_cost < min_insert_cost:
            min_insert_cost = insert_cost
            best_pos = pos
    
    new_route = route[:best_pos] + [most_expensive] + route[best_pos:]
    new_truck_routes = truck_routes.copy()
    new_truck_routes[tid] = new_route
    
    new_drone_assigns = [d.copy() for d in drone_assigns]
    if most_expensive in new_drone_assigns[tid]:
        del new_drone_assigns[tid][most_expensive]
    new_drone_assigns[tid].pop('_intervals', None)
    
    return new_truck_routes, new_drone_assigns


def operator_swap_truck_drone(truck_routes, drone_assigns, tid):
    """
    N3: 交换卡车节点和无人机节点
    """
    route = truck_routes[tid]
    drone_map = drone_assigns[tid]
    
    if len(route) <= 3 or not drone_map:
        return None
    
    truck_candidates = []
    for idx in range(1, len(route)-1):
        cust = route[idx]
        if cust == DEPOT:
            continue
        already_drone = False
        for dm in drone_assigns:
            if cust in dm:
                already_drone = True
                break
        if not already_drone:
            truck_candidates.append((idx, cust))
    
    if not truck_candidates:
        return None
    
    drone_candidates = [c for c in drone_map.keys() if c != '_intervals']
    if not drone_candidates:
        return None
    
    truck_idx, truck_cust = random.choice(truck_candidates)
    drone_cust = random.choice(drone_candidates)
    
    # 获取原无人机任务的起降点
    drone_launch, drone_land = drone_map[drone_cust]
    
    new_route = route.copy()
    new_route[truck_idx] = drone_cust
    
    new_drone_assigns = [d.copy() for d in drone_assigns]
    del new_drone_assigns[tid][drone_cust]
    new_drone_assigns[tid][truck_cust] = (drone_launch, drone_land)
    new_drone_assigns[tid].pop('_intervals', None)
    
    new_truck_routes = truck_routes.copy()
    new_truck_routes[tid] = new_route
    
    return new_truck_routes, new_drone_assigns


def operator_2opt(truck_routes, tid):
    """
    N4: 2-opt优化卡车路径
    """
    route = truck_routes[tid]
    if len(route) <= 3:
        return None
    
    improved = True
    new_route = route.copy()
    
    while improved:
        improved = False
        for i in range(1, len(new_route)-2):
            for j in range(i+1, len(new_route)-1):
                if j - i == 1:
                    continue
                curr_len = dist_mat[new_route[i-1], new_route[i]] + dist_mat[new_route[j], new_route[j+1]]
                new_len = dist_mat[new_route[i-1], new_route[j]] + dist_mat[new_route[i], new_route[j+1]]
                if new_len < curr_len - 1e-6:
                    new_route = new_route[:i] + new_route[i:j+1][::-1] + new_route[j+1:]
                    improved = True
                    break
            if improved:
                break
    
    new_truck_routes = truck_routes.copy()
    new_truck_routes[tid] = new_route
    return new_truck_routes


def operator_drone_reassign(truck_routes, drone_assigns, tid):
    """
    N5: 删除并重新分配最昂贵的无人机任务
    """
    drone_map = drone_assigns[tid]
    if not drone_map:
        return None
    
    most_expensive = None
    max_cost = -inf
    
    for cust, (launch, land) in drone_map.items():
        if cust == '_intervals':
            continue
        cost = dist_mat[launch, cust] + dist_mat[cust, land]
        if cost > max_cost:
            max_cost = cost
            most_expensive = cust
    
    if most_expensive is None:
        return None
    
    # 删除该任务
    new_drone_assigns = [d.copy() for d in drone_assigns]
    del new_drone_assigns[tid][most_expensive]
    new_drone_assigns[tid].pop('_intervals', None)
    
    # 尝试重新分配
    route = truck_routes[tid]
    truck_time = {DEPOT: 0.0}
    for k in range(len(route)-1):
        i, j = route[k], route[k+1]
        truck_time[j] = truck_time[i] + dist_mat[i, j] / TRUCK_SPEED
    
    best_launch = None
    best_land = None
    best_cost = inf
    
    for i_idx in range(len(route)-1):
        launch = route[i_idx]
        if launch == DEPOT:
            continue
        for k_idx in range(i_idx+1, len(route)):
            land = route[k_idx]
            if land == DEPOT:
                continue
            
            d_total = dist_mat[launch, most_expensive] + dist_mat[most_expensive, land]
            if d_total > DRONE_MAX_RANGE:
                continue
            
            launch_time = truck_time.get(launch, 0)
            return_time = launch_time + s_l + d_total / DRONE_SPEED + s_r
            if return_time > truck_time.get(land, inf) + 1e-6:
                continue
            
            if d_total < best_cost:
                best_cost = d_total
                best_launch = launch
                best_land = land
    
    if best_launch is not None:
        new_drone_assigns[tid][most_expensive] = (best_launch, best_land)
    
    return truck_routes, new_drone_assigns


# ====================== 帕累托局部搜索 (PLS) ======================

def pareto_local_search(archive, full_solutions, max_iter=3):
    """
    帕累托局部搜索：对帕累托档案中的解进行邻域搜索
    """
    if len(archive) <= 1:
        return archive, full_solutions
    
    new_archive = archive.copy()
    new_solutions = full_solutions.copy()
    
    search_points = sorted(archive, key=lambda x: x[0] + x[1])[:5]
    
    for point in search_points:
        c, t = point
        key = (round(c, 4), round(t, 4))
        
        if key not in new_solutions:
            continue
        
        routes, drone_assigns, edges, _ = new_solutions[key]
        current_cost, current_delay = c, t
        
        for _ in range(max_iter):
            
            # ===== 第一步：使用 operator_2opt_enhanced 对所有卡车执行2-opt =====
            for tid in range(TRUCK_NUM):
                # 调用增强版2-opt
                new_routes = operator_2opt_enhanced(routes, tid, max_iter=10)
                
                if new_routes is None:
                    continue
                
                # 评估新解
                new_cost, new_delay, new_edges, new_drone_edges, feasible = evaluate_solution(
                    new_routes, drone_assigns, set()
                )
                
                if feasible and new_cost != inf:
                    new_archive = update_archive(new_archive, (new_cost, new_delay))
                    new_key = (round(new_cost, 4), round(new_delay, 4))
                    if new_key not in new_solutions:
                        new_solutions[new_key] = (new_routes, drone_assigns, new_edges, new_drone_edges)
                    
                    if new_cost + new_delay < current_cost + current_delay:
                        current_cost, current_delay = new_cost, new_delay
                        routes = new_routes
            
            # ===== 第二步：无人机任务调整 =====
            for tid in range(TRUCK_NUM):
                operators = [
                    ('truck_to_drone', operator_truck_to_drone, True),
                    ('drone_to_truck', operator_drone_to_truck, True),
                    ('drone_reassign', operator_drone_reassign, True),
                    ('swap', operator_swap_truck_drone, True)
                ]
                
                random.shuffle(operators)
                
                for op_name, op_func, need_drone in operators:
                    try:
                        result = op_func(routes, drone_assigns, tid)
                        if result is None:
                            continue
                        new_routes, new_drone = result
                        
                        truck_cust = set()
                        for rt in new_routes:
                            for n in rt:
                                if n != DEPOT:
                                    truck_cust.add(n)
                        drone_cust = set()
                        for dm in new_drone:
                            for k in dm:
                                if k != '_intervals':
                                    drone_cust.add(k)
                        if truck_cust & drone_cust:
                            continue
                        
                        all_cust = set(range(CUST_START, CUST_END+1))
                        if (truck_cust | drone_cust) != all_cust:
                            continue
                        
                        new_cost, new_delay, new_edges, new_drone_edges, feasible = evaluate_solution(
                            new_routes, new_drone, set()
                        )
                        
                        if not feasible or new_cost == inf:
                            continue
                        
                        new_archive = update_archive(new_archive, (new_cost, new_delay))
                        new_key = (round(new_cost, 4), round(new_delay, 4))
                        if new_key not in new_solutions:
                            new_solutions[new_key] = (new_routes, new_drone, new_edges, new_drone_edges)
                        
                        if new_cost + new_delay < current_cost + current_delay:
                            current_cost, current_delay = new_cost, new_delay
                            routes, drone_assigns = new_routes, new_drone
                            break
                    except Exception as e:
                        continue
    
    return new_archive, new_solutions


# ====================== 增强版 2-opt 算子 ======================

def operator_2opt_enhanced(truck_routes, tid, max_iter=20):
    """
    增强版 2-opt：对单条路径反复优化直到无改进
    """
    route = truck_routes[tid]
    if len(route) <= 3:
        return None
    
    improved = True
    iteration = 0
    new_route = route.copy()
    
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        
        for i in range(1, len(new_route)-2):
            for j in range(i+1, len(new_route)-1):
                if j - i == 1:
                    continue
                
                # 计算当前距离
                curr_len = dist_mat[new_route[i-1], new_route[i]] + dist_mat[new_route[j], new_route[j+1]]
                new_len = dist_mat[new_route[i-1], new_route[j]] + dist_mat[new_route[i], new_route[j+1]]
                
                if new_len < curr_len - 1e-6:
                    new_route = new_route[:i] + new_route[i:j+1][::-1] + new_route[j+1:]
                    improved = True
                    break
            if improved:
                break
    
    if new_route != route:
        new_truck_routes = truck_routes.copy()
        new_truck_routes[tid] = new_route
        return new_truck_routes
    else:
        return None


# ====================== 主优化流程 ======================

def paco_optimize():
    start_time = time.time()
    
    n_node = len(nodes)
    tau_truck = np.full((n_node, n_node), tau0, dtype=np.float64)
    tau_tard_truck = np.full((n_node, n_node), tau0, dtype=np.float64)
    tau_drone = np.full((n_node, n_node, n_node), tau0, dtype=np.float64)
    tau_tard_drone = np.full((n_node, n_node, n_node), tau0, dtype=np.float64)

    pareto_archive = []
    conv_curve = []
    best_sum = inf
    best_feasible = None
    full_solutions = {}

    for it in range(MAX_IT):
        group_sols = [[] for _ in range(PARALLEL_GROUP)]
        ant_per_group = K // PARALLEL_GROUP
        
        modes = []
        for _ in range(K // 5):
            modes.append('delay_first')
            modes.append('cost_first')
            modes.append('balance')
            modes.append('delay_favor')
            modes.append('cost_favor')
        while len(modes) < K:
            modes.append(random.choice(['delay_first', 'cost_first', 'balance', 'delay_favor', 'cost_favor']))
        random.shuffle(modes)
        
        for g in range(PARALLEL_GROUP):
            for idx in range(ant_per_group):
                mode_idx = g * ant_per_group + idx
                if mode_idx < len(modes):
                    mode = modes[mode_idx]
                else:
                    mode = random.choice(['delay_first', 'cost_first', 'balance', 'delay_favor', 'cost_favor'])
                
                c, t, rt, da, te, de, ok, used_mode = ant_construct(
                    tau_truck, tau_tard_truck, tau_drone, tau_tard_drone, 
                    mode=mode
                )
                if ok:
                    group_sols[g].append((c, t, rt, da, te, de, used_mode))

        all_feasible = []
        for g in group_sols:
            all_feasible.extend(g)
        
        for sol in all_feasible:
            c, t, rt, da, te, de, _ = sol
            # pareto_archive = update_archive(pareto_archive, (c, t))
            pareto_archive = update_archive(pareto_archive, (c, t), similarity_threshold=0.5)
            key = (round(c, 4), round(t, 4))
            full_solutions[key] = (rt, da, te, de)

        if all_feasible:
            cur_best = min(all_feasible, key=lambda x: x[0] + x[1])
            if best_feasible is None or (cur_best[0] + cur_best[1]) < (best_feasible[0] + best_feasible[1]):
                best_feasible = cur_best

        if len(pareto_archive) > 0:
            cur_min = min(c + t for c, t in pareto_archive)
            if cur_min < best_sum:
                best_sum = cur_min
        conv_curve.append(best_sum)

        # ===== 触发帕累托局部搜索 =====
        if (it + 1) % PLS_INTERVAL == 0 and len(pareto_archive) > 1:
            old_size = len(pareto_archive)
            pareto_archive, full_solutions = pareto_local_search(
                pareto_archive, 
                full_solutions, 
                max_iter=PLS_MAX_ITER
            )
            if len(pareto_archive) > old_size:
                print(f"  → PLS (Iter {it+1}): {old_size} → {len(pareto_archive)} 个点")
            
            # 更新best_feasible
            if pareto_archive:
                best_point = min(pareto_archive, key=lambda x: x[0] + x[1])
                key = (round(best_point[0], 4), round(best_point[1], 4))
                if key in full_solutions:
                    rt, da, te, de = full_solutions[key]
                    best_feasible = (best_point[0], best_point[1], rt, da, te, de, None)

        # 信息素更新
        tau_truck *= (1 - RHO)
        tau_tard_truck *= (1 - RHO)
        tau_drone *= (1 - RHO)
        tau_tard_drone *= (1 - RHO)

        if best_feasible is not None:
            best_cost, best_delay, best_routes, best_drone_assigns, _, _, _ = best_feasible
            best_delta = Qc / (best_cost + 1e-6) + Qt / (best_delay + 1e-6)
            
            for rt in best_routes:
                for k in range(len(rt)-1):
                    u, v = int(rt[k]), int(rt[k+1])
                    tau_truck[u, v] += best_delta
                    tau_tard_truck[u, v] += best_delta
            
            for da in best_drone_assigns:
                for cust, (launch, land) in da.items():
                    if cust != '_intervals':
                        tau_drone[launch, cust, land] += best_delta
                        tau_tard_drone[launch, cust, land] += best_delta

        # 帕累托解贡献辅助信息素
        if len(pareto_archive) > 1:
            sorted_pareto = sorted(pareto_archive, key=lambda x: x[0] + x[1])
            for idx, (cost, tard) in enumerate(sorted_pareto[:5]):
                weight = 1.0 - idx * 0.15
                delta = (Qc / (cost + 1e-6) + Qt / (tard + 1e-6)) * weight
                key = (round(cost, 4), round(tard, 4))
                if key in full_solutions:
                    rt, da, te, de = full_solutions[key]
                    for route in rt:
                        for k in range(len(route)-1):
                            u, v = int(route[k]), int(route[k+1])
                            tau_truck[u, v] += delta * 0.3
                            tau_tard_truck[u, v] += delta * 0.3
                    for dm in da:
                        for cust, (launch, land) in dm.items():
                            if cust != '_intervals':
                                tau_drone[launch, cust, land] += delta * 0.3
                                tau_tard_drone[launch, cust, land] += delta * 0.3

        tau_truck = np.clip(tau_truck, tau_min, tau_max)
        tau_tard_truck = np.clip(tau_tard_truck, tau_min, tau_max)
        tau_drone = np.clip(tau_drone, tau_min, tau_max)
        tau_tard_drone = np.clip(tau_tard_drone, tau_min, tau_max)

        if (it+1) % 10 == 0:
            print(f"Iter {it+1:3d} | Archive: {len(pareto_archive)} | Best: {best_sum:.2f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if not pareto_archive or best_feasible is None:
        raise RuntimeError("无可行解")
    
    # 从帕累托档案中选出三个极端解
    best_combined = min(pareto_archive, key=lambda x: x[0] + x[1])
    best_cost = min(pareto_archive, key=lambda x: x[0])
    best_delay = min(pareto_archive, key=lambda x: x[1])
    
    def get_full_solution(point):
        c, t = point
        key = (round(c, 4), round(t, 4))
        if key in full_solutions:
            return full_solutions[key]
        min_dist = inf
        best_key = None
        for k in full_solutions.keys():
            dist = abs(k[0] - c) + abs(k[1] - t)
            if dist < min_dist:
                min_dist = dist
                best_key = k
        if best_key and min_dist < 0.1:
            return full_solutions[best_key]
        return None, None, None, None
    
    combined_routes, combined_drone, combined_te, combined_de = get_full_solution(best_combined)
    cost_routes, cost_drone, cost_te, cost_de = get_full_solution(best_cost)
    delay_routes, delay_drone, delay_te, delay_de = get_full_solution(best_delay)
    
    if combined_routes is None:
        combined_routes, combined_drone, combined_te, combined_de = best_feasible[2], best_feasible[3], best_feasible[4], best_feasible[5]
    if cost_routes is None:
        cost_routes, cost_drone, cost_te, cost_de = best_feasible[2], best_feasible[3], best_feasible[4], best_feasible[5]
    if delay_routes is None:
        delay_routes, delay_drone, delay_te, delay_de = best_feasible[2], best_feasible[3], best_feasible[4], best_feasible[5]
    
    result = {
        'combined': {
            'cost': best_combined[0],
            'delay': best_combined[1],
            'routes': combined_routes,
            'drone_assigns': combined_drone,
            'truck_edges': combined_te,
            'drone_edges': combined_de
        },
        'cost_min': {
            'cost': best_cost[0],
            'delay': best_cost[1],
            'routes': cost_routes,
            'drone_assigns': cost_drone,
            'truck_edges': cost_te,
            'drone_edges': cost_de
        },
        'delay_min': {
            'cost': best_delay[0],
            'delay': best_delay[1],
            'routes': delay_routes,
            'drone_assigns': delay_drone,
            'truck_edges': delay_te,
            'drone_edges': delay_de
        },
        'archive': pareto_archive,
        'conv_curve': conv_curve,
        'elapsed_time': elapsed_time
    }
    
    return result


# ====================== 保存和绘图 ======================

def save_results_to_file(result):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("results", f"P-ACO_PLS_result_{timestamp}.txt")
        
        if not os.path.exists("results"):
            os.makedirs("results")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("P-ACO + PLS 混合优化结果\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write("【运行信息】\n")
            f.write(f"  运行时间: {result['elapsed_time']:.2f} 秒\n")
            f.write(f"  客户数量: {CUSTOMER_NUM}\n")
            f.write(f"  卡车数量: {TRUCK_NUM}\n")
            f.write(f"  无人机数量: {DRONE_NUM}\n")
            f.write(f"  无人机续航: {ENDURANCE}\n")
            f.write(f"  最大迭代: {MAX_IT}\n")
            f.write(f"  蚂蚁数量: {K}\n")
            f.write(f"  PLS间隔: 每{PLS_INTERVAL}代\n")
            f.write(f"  帕累托档案大小: {len(result['archive'])}\n\n")
            
            f.write("【三个极端解对比】\n")
            f.write("  " + "-"*50 + "\n")
            f.write(f"  类型      | 成本   | 延迟   | 综合\n")
            f.write("  " + "-"*50 + "\n")
            f.write(f"  综合最小  | {result['combined']['cost']:6.2f} | {result['combined']['delay']:6.2f} | {result['combined']['cost'] + result['combined']['delay']:6.2f}\n")
            f.write(f"  成本最小  | {result['cost_min']['cost']:6.2f} | {result['cost_min']['delay']:6.2f} | {result['cost_min']['cost'] + result['cost_min']['delay']:6.2f}\n")
            f.write(f"  延迟最小  | {result['delay_min']['cost']:6.2f} | {result['delay_min']['delay']:6.2f} | {result['delay_min']['cost'] + result['delay_min']['delay']:6.2f}\n")
            f.write("  " + "-"*50 + "\n\n")
            
            f.write("【综合最小解】\n")
            f.write(f"  总成本: {result['combined']['cost']:.2f}\n")
            f.write(f"  总延迟: {result['combined']['delay']:.2f}\n")
            f.write("  卡车路径:\n")
            for i, route in enumerate(result['combined']['routes']):
                f.write(f"    卡车 {i+1}: {route}\n")
            f.write("  无人机分配:\n")
            for i, drone in enumerate(result['combined']['drone_assigns']):
                clean = {k: v for k, v in drone.items() if k != '_intervals'}
                if clean:
                    f.write(f"    卡车{i+1}的无人机: {clean}\n")
                else:
                    f.write(f"    卡车{i+1}的无人机: 无任务\n")
            
            f.write("\n【帕累托前沿】\n")
            for i, (c, t) in enumerate(result['archive']):
                f.write(f"  解 {i+1}: 成本={c:.2f}, 延迟={t:.2f}, 综合={c+t:.2f}\n")
            
            f.write("\n【收敛曲线】\n")
            valid_conv = [v for v in result['conv_curve'] if v != inf]
            if valid_conv:
                f.write(f"  最终最优值: {valid_conv[-1]:.2f}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("优化完成！\n")
        
        print(f"✅ 详细结果已保存: {filename}")
        return filename
    except Exception as e:
        print(f"❌ 保存详细结果失败: {e}")
        return None


def plot_results(result, save_dir="results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(save_dir, f"P-ACO_PLS_result_{timestamp}.png")
    
    print(f"\n正在生成图表...")
    
    best = result['combined']
    
    plt.rcParams['figure.figsize'] = (16, 5)
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.scatter(nodes[0,0], nodes[0,1], c='red', s=200, marker='*', label='Depot', zorder=5)
    ax1.scatter(nodes[1:,0], nodes[1:,1], c='steelblue', s=70, label='Customer', zorder=3)

    truck_colors = plt.cm.tab10(np.linspace(0, 1, TRUCK_NUM))
    
    drone_cust_all = set()
    for dm in best['drone_assigns']:
        for key in dm:
            if key != '_intervals':
                drone_cust_all.add(key)

    truck_labels_added = set()
    for i, j, tid in best['truck_edges']:
        if i in drone_cust_all and j in drone_cust_all:
            continue
        color = truck_colors[tid % len(truck_colors)]
        if tid not in truck_labels_added:
            label = f'Truck {tid+1}'
            truck_labels_added.add(tid)
        else:
            label = ""
        ax1.plot([nodes[i,0], nodes[j,0]], [nodes[i,1], nodes[j,1]], 
                c=color, lw=2, label=label)

    drone_labels_added = set()
    drawn = set()
    for i, j, tid in best['drone_edges']:
        key = (tid, tuple(sorted((i, j))))
        if key in drawn:
            continue
        drawn.add(key)
        color = truck_colors[tid % len(truck_colors)]
        if tid not in drone_labels_added:
            label = f'Drone T{tid+1}'
            drone_labels_added.add(tid)
        else:
            label = ""
        ax1.plot([nodes[i,0], nodes[j,0]], [nodes[i,1], nodes[j,1]], 
                '--', c=color, lw=1.5, alpha=0.8, label=label)

    for cust in drone_cust_all:
        ax1.scatter(nodes[cust,0], nodes[cust,1], c='gold', s=100, marker='s', 
                   zorder=4, edgecolors='orange', linewidth=2)

    ax1.set_title(f'Combined Best: Cost={best["cost"]:.1f}, Delay={best["delay"]:.1f}', fontsize=10)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.axis('equal')

    conv = result['conv_curve']
    if conv:
        valid_conv = [v for v in conv if v != inf]
        if valid_conv:
            ax2.plot(range(len(valid_conv)), valid_conv, c='red', lw=2)
            ax2.fill_between(range(len(valid_conv)), 0, valid_conv, alpha=0.2, color='red')
    ax2.set_title('Convergence Curve', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Objective Value')
    ax2.grid(alpha=0.3)

    archive = result['archive']
    if archive:
        costs = [item[0] for item in archive]
        tards = [item[1] for item in archive]
        ax3.scatter(costs, tards, c='green', s=60, alpha=0.6, edgecolors='white', linewidth=1)
        
        c1, t1 = result['combined']['cost'], result['combined']['delay']
        ax3.scatter(c1, t1, c='red', s=150, marker='*', zorder=5, 
                   label=f'Combined Best ({c1:.1f}, {t1:.1f})')
        
        c2, t2 = result['cost_min']['cost'], result['cost_min']['delay']
        ax3.scatter(c2, t2, c='blue', s=120, marker='s', zorder=5,
                   label=f'Cost Min ({c2:.1f}, {t2:.1f})')
        
        c3, t3 = result['delay_min']['cost'], result['delay_min']['delay']
        ax3.scatter(c3, t3, c='orange', s=120, marker='^', zorder=5,
                   label=f'Delay Min ({c3:.1f}, {t3:.1f})')
        
        ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax3.set_title(f'Pareto Front ({len(archive)} points)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Total Cost')
        ax3.set_ylabel('Total Penalty')
        ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.show()
    return png_path

# ====================== 批量运行接口 ======================
def run_algorithm(scale, seed, endurance, e_window, l_window):
    """
    批量运行接口 - 供 run_all.py 调用
    """
    import numpy as np
    import random
    import time
    
    # 设置全局参数
    global CUSTOMER_NUM, TRUCK_NUM, DRONE_NUM, ENDURANCE, DRONE_MAX_RANGE
    global K, MAX_IT, E_WINDOW, L_WINDOW, nodes, dist_mat, demand_list, e_list, l_list
    global CUST_START, CUST_END  # 添加
    
    CUSTOMER_NUM = scale
    ENDURANCE = endurance
    DRONE_MAX_RANGE = endurance
    E_WINDOW = e_window
    L_WINDOW = l_window
    
    # 根据规模设置卡车数
    if scale <= 25:
        TRUCK_NUM = 2
        K = 30
        MAX_IT = 100
    elif scale <= 50:
        TRUCK_NUM = 4
        K = 70
        MAX_IT = 170
    else:
        TRUCK_NUM = 8
        K = 120
        MAX_IT = 250
    DRONE_NUM = TRUCK_NUM
    
    # 设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    
    # 生成实例（使用传入的时间窗）
    nodes, dist_mat, demand_list, e_list, l_list = build_instance_with_params(
        scale, e_window, l_window
    )

    CUST_START = 1
    CUST_END = scale
    
    # 运行优化
    start = time.time()
    result = paco_optimize()
    elapsed = time.time() - start
    
    # 提取结果
    best_combined = min(result['archive'], key=lambda x: x[0] + x[1])
    best_cost = min(result['archive'], key=lambda x: x[0])
    best_delay = min(result['archive'], key=lambda x: x[1])
    
    # return {
    #     'scale': scale,
    #     'seed': seed,
    #     'endurance': endurance,
    #     'best_cost': best_combined[0],
    #     'best_delay': best_combined[1],
    #     'best_objective': best_combined[0] + best_combined[1],
    #     'pareto_size': len(result['archive']),
    #     'runtime': elapsed,
    #     'min_cost': best_cost[0],
    #     'min_delay': best_delay[1],
    # }

    return {
        'scale': scale,
        'seed': seed,
        'endurance': endurance,

        # ===== 综合最小解 =====
        'combined_cost': best_combined[0],
        'combined_delay': best_combined[1],
        'combined_objective': best_combined[0] + best_combined[1],
        
        # ===== 成本最小解 =====
        'cost_min_cost': best_cost[0],
        'cost_min_delay': best_cost[1],
        'cost_min_objective': best_cost[0] + best_cost[1],
        
        # ===== 延迟最小解 =====
        'delay_min_cost': best_delay[0],
        'delay_min_delay': best_delay[1],
        'delay_min_objective': best_delay[0] + best_delay[1],
        
        # ===== 其他 =====
        'pareto_size': len(result['archive']),
        'runtime': elapsed,
    }


def build_instance_with_params(cust_n, e_window, l_window):
    """带参数的算例生成"""
    # 根据规模确定坐标范围
    if cust_n <= 25:
        scale = 5
    elif cust_n <= 50:
        scale = 10
    else:
        scale = 15
    
    depot = np.array([[0, 0]])
    custs = np.random.uniform(-scale, scale, (cust_n, 2))
    nodes = np.vstack([depot, custs])
    n = len(nodes)
    dist = np.linalg.norm(nodes[:, None] - nodes[None, :], axis=-1)
    demand = np.random.randint(1, 10, size=cust_n)
    e = np.random.randint(e_window[0], e_window[1], cust_n)
    l = np.random.randint(l_window[0], l_window[1], cust_n)
    return nodes, dist, demand, e, l

# ====================== 运行 ======================

if __name__ == "__main__":
    result = paco_optimize()
    
    print("\n" + "="*60)
    print("三个极端解对比")
    print("="*60)
    print(f"  类型      | 成本   | 延迟   | 综合")
    print("  " + "-"*40)
    print(f"  综合最小  | {result['combined']['cost']:6.2f} | {result['combined']['delay']:6.2f} | {result['combined']['cost'] + result['combined']['delay']:6.2f}")
    print(f"  成本最小  | {result['cost_min']['cost']:6.2f} | {result['cost_min']['delay']:6.2f} | {result['cost_min']['cost'] + result['cost_min']['delay']:6.2f}")
    print(f"  延迟最小  | {result['delay_min']['cost']:6.2f} | {result['delay_min']['delay']:6.2f} | {result['delay_min']['cost'] + result['delay_min']['delay']:6.2f}")
    print("  " + "-"*40)
    print(f"⏱️  运行时间: {result['elapsed_time']:.2f} 秒")
    print(f"📊 帕累托前沿点数: {len(result['archive'])}")
    
    txt_path = save_results_to_file(result)
    png_path = plot_results(result)
    
    print("\n" + "="*60)
    print("✅ 所有结果已保存！")
    if txt_path:
        print(f"  📄 文本: {txt_path}")
    if png_path:
        print(f"  📊 图片: {png_path}")
    print(f"  📊 帕累托前沿点数: {len(result['archive'])}")
    print(f"  ⏱️  运行时间: {result['elapsed_time']:.2f} 秒")
    print("="*60)
