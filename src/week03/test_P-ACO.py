import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from math import inf
from datetime import datetime
import time  # 【新增】导入time模块

# ====================== 论文固定参数 ======================
TRUCK_SPEED = 30
DRONE_SPEED = 60
ENDURANCE = 4
DRONE_MAX_RANGE = ENDURANCE * DRONE_SPEED
DEPOT = 0

# # 25客户
# TRUCK_NUM = 2
# DRONE_NUM = 2
# CUSTOMER_NUM = 25
# K = 30
# MAX_IT = 100

# 50客户
TRUCK_NUM = 4
DRONE_NUM = 4
CUSTOMER_NUM = 50
K = 70
MAX_IT = 170

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
MAX_EAN = 50
PARALLEL_GROUP = 4

E_WINDOW = [5, 30]
L_WINDOW = [60, 110]
TRUCK_CAP = 500
DRONE_CAP = 500
s_l = 0.2
s_r = 0.2
beta_drone = 1.0
M = 10000

np.random.seed(42)
random.seed(42)

# ====================== 1. 算例生成 ======================
def build_instance(cust_n):
    depot = np.array([[0, 0]])
    custs = np.random.uniform(-15, 15, (cust_n, 2))
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

# ====================== 2. 帕累托支配 ======================
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
    new_arch = []
    dominated_flag = False
    for sol in archive:
        if dominates(new_sol, sol):
            continue
        if dominates(sol, new_sol):
            dominated_flag = True
        new_arch.append(sol)
    if dominated_flag:
        return new_arch
    new_arch.append(new_sol)
    if len(new_arch) > MAX_EAN:
        cd = crowding_distance(new_arch)
        min_idx = np.argmin(cd)
        new_arch.pop(min_idx)
    return new_arch

# ====================== 3. 评估函数 ======================
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

        for cust, val in drone_map.items():
            if cust == '_intervals':
                continue
            launch, land = val
            d1 = dist_mat[launch, cust]
            d2 = dist_mat[cust, land]
            total = d1 + d2
            if total > DRONE_MAX_RANGE + 1e-6:
                return inf, inf, [], [], False
            total_drone_cost += total
            drone_edges.append((launch, cust, tid))
            drone_edges.append((cust, land, tid))
            
            arrival_at_cust = truck_time[launch] + s_l + d1 / DRONE_SPEED
            early = max(0.0, e_list[cust-1] - arrival_at_cust)
            late = max(0.0, arrival_at_cust - l_list[cust-1])
            total_tard += early + late

    total_cost = total_truck_cost + total_drone_cost
    return total_cost, total_tard, truck_edges, drone_edges, True


# ====================== 4. 蚂蚁构造 ======================
def ant_construct(tau_truck, tau_tard_truck, tau_drone, tau_tard_drone):
    unvisited = set(range(CUST_START, CUST_END+1))
    
    truck_routes = [[DEPOT] for _ in range(TRUCK_NUM)]
    truck_time = {i: 0.0 for i in range(TRUCK_NUM)}
    drone_assigns = [dict() for _ in range(TRUCK_NUM)]

    # ========== Stage 1: 卡车路径构造 ==========
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

    # 返回仓库
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

    # ========== Stage 2: 无人机分配 ==========
    global_drone_customers = set()
    max_drone_per_truck = 2
    
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

                    truck_dist = dist_mat[i, cust] + dist_mat[cust, k]
                    savings = truck_dist - d_total
                    
                    original_st = raw_truck_time[cust]
                    drone_st = raw_truck_time[i] + s_l + dist_mat[i, cust] / DRONE_SPEED
                    time_improvement = max(0, original_st - drone_st)
                    
                    score = savings + time_improvement * 0.5
                    
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

        # 重建卡车路径
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
    return cost, tard, truck_routes, drone_assigns, te, de, feasible

# ====================== 5. 主流程 ======================
def paco_optimize():
    """P-ACO主优化流程"""
    # 【新增】记录开始时间
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

    for it in range(MAX_IT):
        group_sols = [[] for _ in range(PARALLEL_GROUP)]
        ant_per_group = K // PARALLEL_GROUP
        for g in range(PARALLEL_GROUP):
            for _ in range(ant_per_group):
                c, t, rt, da, te, de, ok = ant_construct(tau_truck, tau_tard_truck, tau_drone, tau_tard_drone)
                if ok:
                    group_sols[g].append((c, t, rt, da, te, de))

        all_feasible = []
        for g in group_sols:
            all_feasible.extend(g)
        for sol in all_feasible:
            pareto_archive = update_archive(pareto_archive, (sol[0], sol[1]))

        if all_feasible:
            cur_best = min(all_feasible, key=lambda x: x[0] + x[1])
            if best_feasible is None or (cur_best[0] + cur_best[1]) < (best_feasible[0] + best_feasible[1]):
                best_feasible = cur_best

        if len(pareto_archive) > 0:
            cur_min = min(c + t for c, t in pareto_archive)
            if cur_min < best_sum:
                best_sum = cur_min
        conv_curve.append(best_sum)

        tau_truck *= (1 - RHO)
        tau_tard_truck *= (1 - RHO)
        tau_drone *= (1 - RHO)
        tau_tard_drone *= (1 - RHO)

        for cost, tard in pareto_archive:
            delta = Qc / (cost + 1e-6) + Qt / (tard + 1e-6)
            if all_feasible:
                match = min(all_feasible, key=lambda x: abs(x[0]-cost)+abs(x[1]-tard))
                
                for rt in match[2]:
                    for k in range(len(rt)-1):
                        u, v = int(rt[k]), int(rt[k+1])
                        tau_truck[u, v] += delta
                        tau_tard_truck[u, v] += delta
                
                for da in match[3]:
                    for cust, (launch, land) in da.items():
                        if cust != '_intervals':
                            tau_drone[launch, cust, land] += delta
                            tau_tard_drone[launch, cust, land] += delta

        tau_truck = np.clip(tau_truck, tau_min, tau_max)
        tau_tard_truck = np.clip(tau_tard_truck, tau_min, tau_max)
        tau_drone = np.clip(tau_drone, tau_min, tau_max)
        tau_tard_drone = np.clip(tau_tard_drone, tau_min, tau_max)

        if (it+1) % 10 == 0:
            print(f"Iter {it+1:3d} | Archive size: {len(pareto_archive)} | MinSumObj: {best_sum:.2f}")

    # 【新增】计算运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if not pareto_archive or best_feasible is None:
        raise RuntimeError("无可行解")
    
    # 【新增】将运行时间附加到返回结果中
    return best_feasible, conv_curve, pareto_archive, elapsed_time

# ====================== 6. 保存结果到文本文件（修改版） ======================
def save_results_to_file(filename, bc, bt, br, bd, conv, archive, elapsed_time):
    """保存详细结果到文本文件（含运行时间）"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("P-ACO 算法优化结果\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            # 【新增】写入运行时间
            f.write("【运行信息】\n")
            f.write(f"  运行时间: {elapsed_time:.2f} 秒\n")
            f.write(f"  客户数量: {CUSTOMER_NUM}\n")
            f.write(f"  卡车数量: {TRUCK_NUM}\n")
            f.write(f"  无人机数量: {DRONE_NUM}\n")
            f.write(f"  无人机续航: {ENDURANCE}\n")
            f.write(f"  最大迭代: {MAX_IT}\n")
            f.write(f"  蚂蚁数量: {K}\n\n")
            
            f.write("【最优解】\n")
            f.write(f"  总成本: {bc:.2f}\n")
            f.write(f"  总延迟: {bt:.2f}\n")
            f.write(f"  综合目标: {bc + bt:.2f}\n\n")
            
            f.write("【卡车路径】\n")
            for i, route in enumerate(br):
                f.write(f"  卡车 {i+1}: {route}\n")
            
            f.write("\n【无人机分配】\n")
            for i, drone in enumerate(bd):
                clean_drone = {k: v for k, v in drone.items() if k != '_intervals'}
                if clean_drone:
                    f.write(f"  无人机 {i+1}:\n")
                    for cust, (launch, land) in clean_drone.items():
                        f.write(f"    客户 {cust}: 起飞点 {launch} -> 降落点 {land}\n")
                else:
                    f.write(f"  无人机 {i+1}: 无任务\n")
            
            f.write("\n【帕累托前沿】\n")
            for i, (c, t) in enumerate(archive):
                f.write(f"  解 {i+1}: 成本={c:.2f}, 延迟={t:.2f}, 综合={c+t:.2f}\n")
            
            f.write("\n【收敛曲线】\n")
            valid_conv = [v for v in conv if v != inf]
            if valid_conv:
                valid_conv_clean = [float(v) for v in valid_conv]
                f.write(f"  最终最优值: {valid_conv_clean[-1]:.2f}\n")
                f.write(f"  迭代次数: {len(valid_conv_clean)}\n")
                if len(valid_conv_clean) > 20:
                    f.write(f"  前10次迭代: {[round(v, 2) for v in valid_conv_clean[:10]]}\n")
                    f.write(f"  后10次迭代: {[round(v, 2) for v in valid_conv_clean[-10:]]}\n")
                else:
                    f.write(f"  全部值: {[round(v, 2) for v in valid_conv_clean]}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("优化完成！\n")
        
        print(f"✅ 详细结果已保存: {filename}")
        return True
    except Exception as e:
        print(f"❌ 保存详细结果失败: {e}")
        return False

# ====================== 7. 绘图 ======================
def plot_all(truck_edges, drone_edges, conv, archive, drone_assigns, save_dir="results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建目录: {save_dir}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"P-ACO_result_{timestamp}"
    png_path = os.path.join(save_dir, f"{base_filename}.png")
    
    print(f"\n正在生成图表...")
    
    plt.rcParams['figure.figsize'] = (16, 5)
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # ---- 子图1: 路线图 ----
    ax1.scatter(nodes[0,0], nodes[0,1], c='red', s=200, marker='*', label='Depot', zorder=5)
    ax1.scatter(nodes[1:,0], nodes[1:,1], c='steelblue', s=70, label='Customer', zorder=3)

    truck_colors = plt.cm.tab10(np.linspace(0, 1, TRUCK_NUM))
    drone_colors = plt.cm.tab20(np.linspace(0, 1, DRONE_NUM * 2))[DRONE_NUM:]
    
    if not truck_edges and not drone_edges:
        print("⚠️  没有可绘制的边")
        plt.close()
        return png_path

    drone_cust_all = set()
    for dm in drone_assigns:
        for key in dm:
            if key != '_intervals':
                drone_cust_all.add(key)

    for i, j, tid in truck_edges:
        if not (i in drone_cust_all and j in drone_cust_all):
            color = truck_colors[tid % len(truck_colors)]
            label = f'Truck {tid+1}' if (i, j, tid) == truck_edges[0] else ""
            ax1.plot([nodes[i,0], nodes[j,0]], [nodes[i,1], nodes[j,1]], 
                    c=color, lw=2, label=label)

    drawn = set()
    for i, j, tid in drone_edges:
        key = (tid, tuple(sorted((i, j))))
        if key not in drawn:
            drawn.add(key)
            color = drone_colors[tid % len(drone_colors)]
            label = f'Drone {tid+1}' if key == next(iter(drawn)) else ""
            ax1.plot([nodes[i,0], nodes[j,0]], [nodes[i,1], nodes[j,1]], 
                    '--', c=color, lw=1.5, label=label)

    for cust in drone_cust_all:
        ax1.scatter(nodes[cust,0], nodes[cust,1], c='gold', s=100, marker='s', 
                   zorder=4, edgecolors='orange', linewidth=2)
        ax1.annotate(f'D{cust}', (nodes[cust,0]+0.3, nodes[cust,1]+0.3), fontsize=8, color='darkorange')

    for i in range(1, len(nodes)):
        if i not in drone_cust_all:
            ax1.annotate(str(i), (nodes[i,0]+0.2, nodes[i,1]+0.2), fontsize=8, alpha=0.7)

    ax1.set_title('Truck-Drone Delivery Routes', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.axis('equal')

    # ---- 子图2: 收敛曲线 ----
    if conv:
        valid_conv = [v for v in conv if v != inf]
        if valid_conv:
            ax2.plot(range(len(valid_conv)), valid_conv, c='red', lw=2)
            ax2.fill_between(range(len(valid_conv)), 0, valid_conv, alpha=0.2, color='red')
            
            min_val = min(valid_conv)
            max_val = max(valid_conv)
            padding = (max_val - min_val) * 0.1
            ax2.set_ylim(min_val - padding, max_val + padding)
            
    ax2.set_title('Convergence Curve', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Objective Value')
    ax2.grid(alpha=0.3)

    # ---- 子图3: 帕累托前沿 ----
    if archive:
        costs = [p[0] for p in archive]
        tards = [p[1] for p in archive]
        ax3.scatter(costs, tards, c='green', s=60, alpha=0.8, edgecolors='white', linewidth=1)
        ax3.set_title('Pareto Front', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Total Cost')
        ax3.set_ylabel('Total Tardiness')
        ax3.grid(alpha=0.3)

    plt.tight_layout()
    
    try:
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图片已保存: {png_path}")
    except Exception as e:
        print(f"❌ 保存图片失败: {e}")
        fallback_path = "result.png"
        plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图片已保存到备选路径: {fallback_path}")
    
    plt.show()
    return png_path

# ====================== 8. 运行 ======================
if __name__ == "__main__":
    print("="*60)
    print("协同帕累托蚁群算法 (Collaborative P-ACO)")
    print("IEEE Access, Vol. 8, 2020")
    print("="*60)
    print(f"客户数量: {CUSTOMER_NUM}")
    print(f"卡车数量: {TRUCK_NUM}")
    print(f"无人机数量: {DRONE_NUM}")
    print(f"无人机续航: {ENDURANCE}")
    print(f"最大迭代: {MAX_IT}")
    print(f"蚂蚁数量: {K}")
    print("="*60 + "\n")
    
    # 接收运行时间
    (bc, bt, br, bd, bte, bde), conv, arch, elapsed_time = paco_optimize()
    
    print("\n" + "="*60)
    print("最优解信息")
    print("="*60)
    print(f"总成本: {bc:.2f}")
    print(f"总延迟: {bt:.2f}")
    print(f"综合目标: {bc + bt:.2f}")
    print(f"⏱️  运行时间: {elapsed_time:.2f} 秒")  # 【新增】打印运行时间
    print("\n卡车路径:")
    for i, route in enumerate(br):
        print(f"  卡车 {i+1}: {route}")
    print("\n无人机分配:")
    for i, drone in enumerate(bd):
        clean_drone = {k: v for k, v in drone.items() if k != '_intervals'}
        if clean_drone:
            print(f"  无人机 {i+1}:")
            for cust, (launch, land) in clean_drone.items():
                print(f"    客户 {cust}: 起飞点 {launch} -> 降落点 {land}")
        else:
            print(f"  无人机 {i+1}: 无任务")
    
    print("\n" + "="*60)
    print("保存结果到本地...")
    print("="*60)
    
    save_dir = "results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 【修改】保存文本结果时传入运行时间
    txt_path = os.path.join(save_dir, f"P-ACO_result_{timestamp}.txt")
    save_results_to_file(txt_path, bc, bt, br, bd, conv, arch, elapsed_time)
    
    png_path = plot_all(bte, bde, conv, arch, bd, save_dir=save_dir)
    
    print("\n" + "="*60)
    print("✅ 所有结果已保存！")
    print(f"  📊 图片: {png_path}")
    print(f"  📄 文本: {txt_path}")
    print(f"  ⏱️  运行时间: {elapsed_time:.2f} 秒")
    print("="*60)