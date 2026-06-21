import torch
import numpy as np


def get_random_problems(batch_size, problem_size):
    """生成基础 CVRP 问题实例"""
    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 200:
        demand_scaler = 70
    else:
        # 通用公式：线性插值
        demand_scaler = int(30 + (problem_size - 20) * (50 - 30) / (100 - 20))

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand


def get_random_problems_with_ev_tw(batch_size, problem_size, use_ev=False, use_tw=False):
    """
    生成带 EV 和 TW 约束的问题实例。
    
    时间窗生成逻辑参考 Solomon benchmark 风格：
    - 先计算每个节点到 depot 的距离作为基础到达时间
    - 时间窗宽度与问题规模成正比，确保可行解存在
    - horizon（总时间上限）根据问题规模设置
    """
    depot_xy, node_xy, node_demand = get_random_problems(batch_size, problem_size)

    result = {
        'depot_xy': depot_xy,
        'node_xy': node_xy,
        'node_demand': node_demand,
    }

    if use_tw:
        # 计算每个节点到 depot 的距离，作为 earliest possible arrival 的参考
        # depot_xy: (batch, 1, 2), node_xy: (batch, problem, 2)
        dist_to_depot = torch.sqrt(((node_xy - depot_xy) ** 2).sum(dim=2))
        # shape: (batch, problem)

        # 时间 horizon 根据问题规模设置
        # 对于 n=50，路线大约访问 8-10 个节点，总路程约 3-5
        # 对于 n=100，路线大约访问 8-12 个节点，总路程约 4-6
        if problem_size <= 50:
            horizon = 5.0
        elif problem_size <= 100:
            horizon = 8.0
        else:
            horizon = 12.0

        # 服务时间：较小值，与 horizon 成比例
        node_service_time = torch.rand(size=(batch_size, problem_size)) * 0.05 * horizon + 0.01
        # shape: (batch, problem)

        # 时间窗生成：
        # tw_early = 节点到 depot 距离 + 随机偏移（模拟从 depot 出发后的最早到达）
        # tw_width = 随机宽度，确保足够宽以允许灵活调度
        tw_center = dist_to_depot + torch.rand(size=(batch_size, problem_size)) * (horizon * 0.6)
        tw_width = torch.rand(size=(batch_size, problem_size)) * (horizon * 0.3) + horizon * 0.15
        # 确保时间窗足够宽

        tw_early = (tw_center - tw_width / 2).clamp(min=0.0)
        tw_late = (tw_center + tw_width / 2).clamp(max=horizon)

        # 确保 tw_late > tw_early + service_time（可行性保证）
        tw_late = torch.maximum(tw_late, tw_early + node_service_time + 0.1)

        result['node_service_time'] = node_service_time
        result['tw_early'] = tw_early
        result['tw_late'] = tw_late
        result['horizon'] = horizon

    if use_ev:
        # 电池容量设置：确保满电至少能走 horizon 距离的一部分
        ev_params = {
            'battery_capacity': 100.0,
            'energy_consumption_rate': 0.5,  # 每单位距离消耗 0.5 电量
            'charging_speed': 1.0,
        }
        # 满电可行驶距离 = 100 / 0.5 = 200 单位距离，远超单条路线所需
        initial_battery = torch.ones(size=(batch_size, 1)) * ev_params['battery_capacity']
        result['initial_battery'] = initial_battery
        result['ev_params'] = ev_params

    return result


def augment_xy_data_by_8_fold(xy_data):
    """8折对称性数据增强"""
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data
