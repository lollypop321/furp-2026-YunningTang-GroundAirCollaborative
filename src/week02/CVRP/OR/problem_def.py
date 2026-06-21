import torch
import numpy as np
import random


def get_random_problems_with_ev_tw(batch_size, problem_size, use_ev=True, use_tw=True):
    """
    生成 CVRP 问题实例，与 POMO 版本保持一致
    支持 EV 能量约束和时间窗约束
    """
    # 节点坐标 (0-1 范围内均匀分布)
    depot_xy = torch.rand(size=(batch_size, 1, 2))
    node_xy = torch.rand(size=(batch_size, problem_size, 2))

    # 需求：参考 Solomon benchmark 的缩放
    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 200:
        demand_scaler = 70
    else:
        demand_scaler = int(30 + (problem_size - 20) * 40 / 180)

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)).float()
    node_demand = node_demand / float(demand_scaler)

    # EV 参数
    battery_capacity = 100.0
    energy_consumption_rate = 0.5
    initial_battery = torch.ones(size=(batch_size, 1)) * battery_capacity

    # 时间窗参数
    if use_tw:
        # 计算节点到 depot 的距离
        depot_coord = depot_xy.squeeze(1)
        distances_to_depot = torch.sqrt(((node_xy - depot_coord.unsqueeze(1)) ** 2).sum(dim=2))

        # 服务时间
        node_service_time = torch.ones(size=(batch_size, problem_size)) * 0.1

        # 时间 horizon 与问题规模成正比
        if problem_size <= 50:
            horizon = 5.0
        elif problem_size <= 100:
            horizon = 8.0
        else:
            horizon = 12.0

        # tw_early 基于到 depot 的距离 + 随机偏移
        tw_early = distances_to_depot * 0.5 + torch.rand(batch_size, problem_size) * 0.5

        # tw_width 与距离相关（近 depot 的节点时间窗更紧）
        tw_width = 1.0 + distances_to_depot * 0.5 + torch.rand(batch_size, problem_size) * 0.5
        tw_width = torch.clamp(tw_width, min=1.0, max=horizon * 0.5)

        tw_late = tw_early + tw_width

        # 确保时间窗在 horizon 内
        tw_early = torch.clamp(tw_early, min=0.0, max=horizon * 0.7)
        tw_late = torch.clamp(tw_late, min=(tw_early + 0.5).float(), max=torch.tensor(horizon))
    else:
        node_service_time = None
        tw_early = None
        tw_late = None

    return {
        'depot_xy': depot_xy,
        'node_xy': node_xy,
        'node_demand': node_demand,
        'node_service_time': node_service_time,
        'tw_early': tw_early,
        'tw_late': tw_late,
        'initial_battery': initial_battery,
        'battery_capacity': battery_capacity,
        'energy_consumption_rate': energy_consumption_rate,
    }


def augment_xy_data_by_8_fold(xy_data):
    """8折数据增强"""
    x = xy_data[:, :, 0]
    y = xy_data[:, :, 1]

    dat1 = xy_data
    dat2 = torch.stack([1 - x, y], dim=2)
    dat3 = torch.stack([x, 1 - y], dim=2)
    dat4 = torch.stack([1 - x, 1 - y], dim=2)
    dat5 = torch.stack([y, x], dim=2)
    dat6 = torch.stack([1 - y, x], dim=2)
    dat7 = torch.stack([y, 1 - x], dim=2)
    dat8 = torch.stack([1 - y, 1 - x], dim=2)

    aug_xy_data = torch.cat([dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8], dim=0)
    return aug_xy_data
