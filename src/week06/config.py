# ===== config.py =====
# 所有规模配置统一管理
SEEDS = [42, 142, 242, 342, 442]

SCALE_CONFIGS = {
    25: {
        'truck_num': 2,
        'drone_num': 2,
        'endurance': 2,
        'e_window': [0, 3],
        'l_window': [12, 20],
        'k': 30,
        'max_it': 100,
        'coords_scale': 5,
    },
    50: {
        'truck_num': 4,
        'drone_num': 4,
        'endurance': 2,
        'e_window': [0, 5],
        'l_window': [18, 28],
        'k': 70,
        'max_it': 170,
        'coords_scale': 10,
    },
    100: {
        'truck_num': 8,
        'drone_num': 8,
        'endurance': 2,
        'e_window': [0, 6],
        'l_window': [22, 34],
        'k': 120,
        'max_it': 250,
        'coords_scale': 15,
    },
}