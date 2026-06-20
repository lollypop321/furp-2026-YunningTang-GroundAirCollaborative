import torch

import os
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from utils.utils import *


class CVRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.time_estimator = TimeEstimator()
        
        self.use_energy_constraint = env_params.get('use_energy_constraint', False)
        self.use_time_window_constraint = env_params.get('use_time_window_constraint', False)

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        
        tw_violation_AM = AverageMeter() if self.use_time_window_constraint else None
        energy_violation_AM = AverageMeter() if self.use_energy_constraint else None

        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score, constraint_stats = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            
            if constraint_stats is not None:
                if self.use_time_window_constraint and 'tw_violation' in constraint_stats:
                    tw_violation_AM.update(constraint_stats['tw_violation'], batch_size)
                if self.use_energy_constraint and 'energy_violation' in constraint_stats:
                    energy_violation_AM.update(constraint_stats['energy_violation'], batch_size)

            episode += batch_size

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            
            msg = f"episode {episode:3d}/{test_num_episode:3d}, Elapsed[{elapsed_time_str}], Remain[{remain_time_str}], score:{score:.3f}, aug_score:{aug_score:.3f}"
            if self.use_time_window_constraint or self.use_energy_constraint:
                if self.use_time_window_constraint:
                    msg += f", TW_Vio:{tw_violation_AM.avg:.4f}"
                if self.use_energy_constraint:
                    msg += f", E_Vio:{energy_violation_AM.avg:.4f}"
            self.logger.info(msg)

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
                
                if self.use_time_window_constraint or self.use_energy_constraint:
                    self.logger.info(" *** Constraint Violations *** ")
                    if self.use_time_window_constraint:
                        self.logger.info(" TIME_WINDOW VIOLATIONS: {:.4f} ".format(tw_violation_AM.avg))
                    if self.use_energy_constraint:
                        self.logger.info(" ENERGY_SHORTAGE VIOLATIONS: {:.4f} ".format(energy_violation_AM.avg))

    def _test_one_batch(self, batch_size):

        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            state, reward, done = self.env.step(selected)

        constraint_stats = None
        if self.use_time_window_constraint or self.use_energy_constraint:
            constraint_stats = {}
            if self.use_time_window_constraint:
                tw_violations = state.time_window_violation.float().mean().item()
                constraint_stats['tw_violation'] = tw_violations
            if self.use_energy_constraint:
                energy_violations = state.energy_shortage.float().mean().item()
                constraint_stats['energy_violation'] = energy_violations

        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)

        max_pomo_reward, _ = aug_reward.max(dim=2)
        no_aug_score = -max_pomo_reward[0, :].float().mean()

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)
        aug_score = -max_aug_pomo_reward.float().mean()

        return no_aug_score.item(), aug_score.item(), constraint_stats