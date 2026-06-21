##########################################################################################
# Machine Environment Config

import os
import sys

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")


##########################################################################################
# import

import logging
from utils.utils import create_logger

from CVRPTester import CVRPTester as Tester


##########################################################################################
# parameters

# 与 train_n50_with_EV_TW.py 保持一致
env_params = {
    'problem_size': 50,
    'pomo_size': 50,
    'use_energy_constraint': True,
    'use_time_window_constraint': True,
    'ev_params': {
        'battery_capacity': 100.0,
        'energy_consumption_rate': 0.5,
        'charging_speed': 1.0,
    },
    # 测试时使用最大惩罚系数（模拟训练后期的约束强度）
    'tw_penalty_coeff': 5.0,
    'energy_penalty_coeff': 5.0,
}

# 与 train_n50_with_EV_TW.py 保持一致
model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    # 加载训练好的 checkpoint（需要修改 path 和 epoch 为实际值）
    'model_load': {
        'path': './result/20260621_012507_train_cvrp_n50_with_EV_TW',  # ← 改成实际路径
        'epoch': 200,  # ← 改成要测试的 epoch
    },
    'test_episodes': 100,  # 测试 100 个实例（与 GA/OR/ALNS 一致）
    'test_batch_size': 100,
    'augmentation_enable': True,  # 启用 8-fold 数据增强
    'aug_factor': 8,
    'aug_batch_size': 25,  # 增强后 batch_size = 100 / 8 ≈ 13，取整为 25 更保险
    # 加载保存的测试数据（确保与 GA/OR/ALNS 使用相同实例）
    'test_data_load': {
        'enable': True,
        'filename': '../vrp50_test_seed1234.pt'
    },
}

# 如果启用增强，调整 batch_size
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'test_cvrp_n50_with_EV_TW',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    tester.run()


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
