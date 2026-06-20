import torch
import numpy as np


def get_random_problems(batch_size, problem_size):

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
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand


def get_random_problems_with_ev_tw(batch_size, problem_size, use_ev=False, use_tw=False):
    """生成带 EV 和 TW 约束的问题"""
    depot_xy, node_xy, node_demand = get_random_problems(batch_size, problem_size)
    
    result = {
        'depot_xy': depot_xy,
        'node_xy': node_xy,
        'node_demand': node_demand,
    }
    
    if use_tw:
        node_service_time = torch.rand(size=(batch_size, problem_size)) * 0.05 + 0.01
        tw_early = torch.rand(size=(batch_size, problem_size)) * 0.3
        tw_late = tw_early + torch.rand(size=(batch_size, problem_size)) * 0.5 + 0.2
        tw_late = torch.clamp(tw_late, max=1.0)
        
        result['node_service_time'] = node_service_time
        result['tw_early'] = tw_early
        result['tw_late'] = tw_late
    
    if use_ev:
        initial_battery = torch.ones(size=(batch_size, 1)) * 100.0
        result['initial_battery'] = initial_battery
        result['ev_params'] = {
            'battery_capacity': 100.0,
            'energy_consumption_rate': 0.5,
            'charging_speed': 1.0,
        }
    
    return result


def augment_xy_data_by_8_fold(xy_data):
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