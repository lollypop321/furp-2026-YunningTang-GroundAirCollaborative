"""
P-ACO (Pareto Ant Colony Optimization) for Truck-Drone Routing with Time Windows.

This module solves the TDRPTW using a hybrid approach combining:
- P-ACO global search with 5 personality modes (delay_first, cost_first, 
  balance, delay_favor, cost_favor)
- Pareto archive with crowding distance and boundary protection
- Greedy drone assignment with score-based insertion
"""

import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from math import inf
from datetime import datetime
import time

# ====================== Parameter Configuration ======================
TRUCK_SPEED = 1.0
DRONE_SPEED = 2.0
ENDURANCE = 4
DRONE_MAX_RANGE = ENDURANCE
DEPOT = 0

# # 25 customers
# TRUCK_NUM = 2
# DRONE_NUM = 2
# CUSTOMER_NUM = 25
# K = 30
# MAX_IT = 100

# 50 customers
TRUCK_NUM = 4
DRONE_NUM = TRUCK_NUM
CUSTOMER_NUM = 50
K = 70
MAX_IT = 170

# # 100 customers
# TRUCK_NUM = 8
# DRONE_NUM = TRUCK_NUM
# CUSTOMER_NUM = 100
# K = 120
# MAX_IT = 250

# ACO parameters
ALPHA = 1          # Pheromone influence weight
BETA = 2           # Heuristic influence weight
q0 = 0.5           # Exploitation vs exploration probability
RHO = 0.15         # Global pheromone evaporation rate
xi = 0.1           # Local pheromone decay rate (per edge traversal)

# Pheromone deposit constants
Qt = 60            # Tardiness component weight
Qc = 120           # Cost component weight
C = 1000           # Scaling constant for heuristic calculation

# Pheromone bounds
tau0 = 10          # Initial pheromone value
tau_max = 20       # Upper bound (prevents premature convergence)
tau_min = 1        # Lower bound (maintains exploration)

# Archive and parallel settings
MAX_EAN = 80       # Maximum Pareto archive size
PARALLEL_GROUP = 4 # Number of parallel ant groups


# # 25 customers time windows
# E_WINDOW = [0, 3]
# L_WINDOW = [10, 18]

# 50 customers time windows 
E_WINDOW = [0, 5]
L_WINDOW = [18, 28]

# # 100 customers time windows
# E_WINDOW = [0, 6]
# L_WINDOW = [22, 34]

# Maximum drone customers per truck
max_drone_per_truck = 4

# Drone operation parameters
s_l = 0.2
s_r = 0.2
M = 10000

np.random.seed(142)
random.seed(142)

# ====================== Instance Generation ======================
def build_instance(cust_n):
    depot = np.array([[0, 0]])
    if cust_n <= 25:
        scale = 5
    elif cust_n <= 50:
        scale = 10
    elif cust_n <= 100:
        scale = 15
    else:
        scale = 20
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
print("P-ACO Multi-Personality Optimization")
print("="*60)
print(f"Customers: {CUSTOMER_NUM}")
print(f"Trucks: {TRUCK_NUM}")
print(f"Drones: {DRONE_NUM}")
print(f"Drone Endurance: {ENDURANCE}")
print(f"Max Iterations: {MAX_IT}")
print(f"Ants: {K}")
print(f"Personality Modes: delay_first / cost_first / balance / delay_favor / cost_favor")
print(f"Time Windows: E=[{min(e_list)}-{max(e_list)}], L=[{min(l_list)}-{max(l_list)}]")
print("="*60 + "\n")

# ====================== Pareto Dominance ======================
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

def update_archive(archive, new_sol):
    # Update Pareto archive with boundary protection
    if not archive:
        return [new_sol]
    
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
    
    # Boundary protection during pruning
    if len(new_arch) > MAX_EAN:
        cd = crowding_distance(new_arch)
        cost_min_idx = np.argmin([s[0] for s in new_arch])
        delay_min_idx = np.argmin([s[1] for s in new_arch])
        cd[cost_min_idx] = inf
        cd[delay_min_idx] = inf
        min_idx = np.argmin(cd)
        new_arch.pop(min_idx)
    
    return new_arch

# ====================== Evaluation Function ======================
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

        # Sequential scheduling check
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


# ====================== Ant Construction (5 Personality Modes) ======================
def ant_construct(tau_truck, tau_tard_truck, tau_drone, tau_tard_drone, mode=None):
    """
    Five personality modes:
    - delay_first: strongly prioritise delay reduction
    - cost_first: strongly prioritise cost reduction
    - balance: balanced trade-off
    - delay_favor: moderately prioritise delay
    - cost_favor: moderately prioritise cost
    """
    if mode is None:
        mode = random.choice(['delay_first', 'cost_first', 'balance', 'delay_favor', 'cost_favor'])
    
    # Weight configuration for each personality mode
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
    else:  # cost_favor
        delay_weight = 3.0
        cost_weight = 8.0

    n_customers = len(nodes) - 1 
    unvisited = set(range(1, n_customers + 1)) 
    
    truck_routes = [[DEPOT] for _ in range(TRUCK_NUM)]
    truck_time = {i: 0.0 for i in range(TRUCK_NUM)}
    drone_assigns = [dict() for _ in range(TRUCK_NUM)]

    # ========== Stage 1: Truck Route Construction ==========
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

    # Return to depot
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

    # ========== Stage 2: Drone Assignment ==========
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
            
            # Compute truck arrival time at this customer
            truck_arrival_time = 0
            for k in range(cust_idx):
                truck_arrival_time += dist_mat[raw_route[k], raw_route[k+1]] / TRUCK_SPEED
            
            # Compute truck lateness at this customer
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

                    # Compute drone arrival time at customer
                    drone_arrival = raw_truck_time[i] + s_l + dist_mat[i, cust] / DRONE_SPEED
                    
                    # Compute tardiness improvement
                    drone_late = max(0, drone_arrival - l_list[cust-1])
                    tard_improvement = truck_late - drone_late
                    
                    # Compute distance saving
                    truck_dist = dist_mat[i, cust] + dist_mat[cust, k]
                    savings = truck_dist - d_total
                    
                    # Score using personality weights
                    score = tard_improvement * delay_weight + savings * cost_weight
                    
                    # Bonus if drone can avoid lateness
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

        # Reconstruct truck route
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


# ====================== P-ACO Optimizer ======================

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
        
        # 5 personality modes, each 20%
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
            pareto_archive = update_archive(pareto_archive, (c, t))
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

        # Pheromone update (combined objective priority)
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

        # Pareto-assisted pheromone deposition
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
        raise RuntimeError("No feasible solution found")
    
    # Extract three extreme solutions from Pareto archive
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

# ====================== Save and Plot Results ======================
def save_results_to_file(result):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("results", f"P-ACO_result_{timestamp}.txt")
        
        if not os.path.exists("results"):
            os.makedirs("results")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("P-ACO Multi-Personality Optimization Results\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write("[Run Information]\n")
            f.write(f"  Runtime: {result['elapsed_time']:.2f} s\n")
            f.write(f"  Customers: {CUSTOMER_NUM}\n")
            f.write(f"  Trucks: {TRUCK_NUM}\n")
            f.write(f"  Drones: {DRONE_NUM}\n")
            f.write(f"  Drone Endurance: {ENDURANCE}\n")
            f.write(f"  Max Iterations: {MAX_IT}\n")
            f.write(f"  Ants: {K}\n")
            f.write(f"  Personality Modes: delay_first / cost_first / balance / delay_favor / cost_favor\n")
            f.write(f"  Pareto Archive Size: {len(result['archive'])}\n\n")
            
            f.write("[Three Extreme Solutions]\n")
            f.write("  " + "-"*50 + "\n")
            f.write(f"  Type        | Cost   | Delay  | Combined\n")
            f.write("  " + "-"*50 + "\n")
            f.write(f"  Combined Min| {result['combined']['cost']:6.2f} | {result['combined']['delay']:6.2f} | {result['combined']['cost'] + result['combined']['delay']:6.2f}\n")
            f.write(f"  Cost Min    | {result['cost_min']['cost']:6.2f} | {result['cost_min']['delay']:6.2f} | {result['cost_min']['cost'] + result['cost_min']['delay']:6.2f}\n")
            f.write(f"  Delay Min   | {result['delay_min']['cost']:6.2f} | {result['delay_min']['delay']:6.2f} | {result['delay_min']['cost'] + result['delay_min']['delay']:6.2f}\n")
            f.write("  " + "-"*50 + "\n\n")
            
            f.write("[Combined Best Solution]\n")
            f.write(f"  Total Cost: {result['combined']['cost']:.2f}\n")
            f.write(f"  Total Delay: {result['combined']['delay']:.2f}\n")
            f.write("  Truck Routes:\n")
            for i, route in enumerate(result['combined']['routes']):
                f.write(f"    Truck {i+1}: {route}\n")
            f.write("  Drone Assignments:\n")
            for i, drone in enumerate(result['combined']['drone_assigns']):
                clean = {k: v for k, v in drone.items() if k != '_intervals'}
                if clean:
                    f.write(f"    Truck {i+1} Drones: {clean}\n")
                else:
                    f.write(f"    Truck {i+1} Drones: None\n")
            
            f.write("\n[Pareto Front]\n")
            for i, (c, t) in enumerate(result['archive']):
                f.write(f"  Solution {i+1}: Cost={c:.2f}, Delay={t:.2f}, Combined={c+t:.2f}\n")
            
            f.write("\n[Convergence Curve]\n")
            valid_conv = [v for v in result['conv_curve'] if v != inf]
            if valid_conv:
                f.write(f"  Final Best Value: {valid_conv[-1]:.2f}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("Optimization Complete!\n")
        
        print(f"Detailed results saved: {filename}")
        return filename
    except Exception as e:
        print(f"Failed to save results: {e}")
        return None


def plot_results(result, save_dir="results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(save_dir, f"P-ACO_result_{timestamp}.png")

    print("\nGenerating plots...")
    
    best = result['combined']
    
    plt.rcParams['figure.figsize'] = (16, 5)
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # ---- Subplot 1: Route Visualisation ----
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

    # ---- Subplot 2: Convergence Curve ----
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

    # ---- Subplot 3: Pareto Front ----
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
        ax3.scatter(costs, tards, c='green', s=60, alpha=0.8, edgecolors='white', linewidth=1)
        ax3.set_title('Pareto Front', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Total Cost')
        ax3.set_ylabel('Total Penalty')
        ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.show()
    return png_path

# ====================== Batch Run Interface ======================
def run_algorithm(scale, seed, endurance, e_window, l_window):
    """
    Batch run interface for run_all.py
    """
    import numpy as np
    import random
    import time
    
    # Set global parameters
    global CUSTOMER_NUM, TRUCK_NUM, DRONE_NUM, ENDURANCE, DRONE_MAX_RANGE
    global K, MAX_IT, E_WINDOW, L_WINDOW, nodes, dist_mat, demand_list, e_list, l_list
    global CUST_START, CUST_END
    
    CUSTOMER_NUM = scale
    ENDURANCE = endurance
    DRONE_MAX_RANGE = endurance
    E_WINDOW = e_window
    L_WINDOW = l_window
    
    # Set truck count based on scale
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
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate instance with given parameters
    nodes, dist_mat, demand_list, e_list, l_list = build_instance_with_params(
        scale, e_window, l_window
    )

    CUST_START = 1
    CUST_END = scale
    
    # Run optimization
    start = time.time()
    result = paco_optimize()
    elapsed = time.time() - start
    
    # Extract results
    best_combined = min(result['archive'], key=lambda x: x[0] + x[1])
    best_cost = min(result['archive'], key=lambda x: x[0])
    best_delay = min(result['archive'], key=lambda x: x[1])
    
    return {
        'scale': scale,
        'seed': seed,
        'endurance': endurance,

        # Combined best solution
        'combined_cost': best_combined[0],
        'combined_delay': best_combined[1],
        'combined_objective': best_combined[0] + best_combined[1],
        
        # Cost best solution
        'cost_min_cost': best_cost[0],
        'cost_min_delay': best_cost[1],
        'cost_min_objective': best_cost[0] + best_cost[1],
        
        # Delay best solution
        'delay_min_cost': best_delay[0],
        'delay_min_delay': best_delay[1],
        'delay_min_objective': best_delay[0] + best_delay[1],
        
        'pareto_size': len(result['archive']),
        'runtime': elapsed,
    }


def build_instance_with_params(cust_n, e_window, l_window):
    """Generate instance with custom time windows"""
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

# ====================== Main ======================
if __name__ == "__main__":
    result = paco_optimize()
    
    print("\n" + "="*60)
    print("Three Extreme Solutions")
    print("="*60)
    print(f"  Type        | Cost   | Delay  | Combined")
    print("  " + "-"*40)
    print(f"  Combined Min| {result['combined']['cost']:6.2f} | {result['combined']['delay']:6.2f} | {result['combined']['cost'] + result['combined']['delay']:6.2f}")
    print(f"  Cost Min    | {result['cost_min']['cost']:6.2f} | {result['cost_min']['delay']:6.2f} | {result['cost_min']['cost'] + result['cost_min']['delay']:6.2f}")
    print(f"  Delay Min   | {result['delay_min']['cost']:6.2f} | {result['delay_min']['delay']:6.2f} | {result['delay_min']['cost'] + result['delay_min']['delay']:6.2f}")
    print("  " + "-"*40)
    print(f"Runtime: {result['elapsed_time']:.2f} s")
    print(f"Pareto Front Size: {len(result['archive'])}")
    
    txt_path = save_results_to_file(result)
    png_path = plot_results(result)
    
    print("\n" + "="*60)
    print("All results saved!")
    if txt_path:
        print(f"  Text: {txt_path}")
    if png_path:
        print(f"  Plot: {png_path}")
    print(f"  Pareto Front Size: {len(result['archive'])}")
    print(f"  Runtime: {result['elapsed_time']:.2f} s")
    print("="*60)



