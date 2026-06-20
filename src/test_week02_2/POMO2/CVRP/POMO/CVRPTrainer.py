import torch
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *


class CVRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # USE_CUDA = self.trainer_params['use_cuda']
        # if USE_CUDA:
        #     cuda_device_num = self.trainer_params['cuda_device_num']
        #     torch.cuda.set_device(cuda_device_num)
        #     # device = torch.device('cuda', cuda_device_num)
        #     if torch.cuda.is_available():
        #         torch.cuda.set_device(cuda_device_num)
        #     else:
        #         print("当前设备无CUDA，自动使用CPU训练")
        #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # else:
        #     device = torch.device('cpu')
        #     torch.set_default_tensor_type('torch.FloatTensor')

        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA and torch.cuda.is_available():
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            if USE_CUDA:
                print("当前设备无CUDA，自动使用CPU训练")
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor') 

        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        self.time_estimator = TimeEstimator()
        
        self.use_energy_constraint = env_params.get('use_energy_constraint', False)
        self.use_time_window_constraint = env_params.get('use_time_window_constraint', False)

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # 版本2删
            # self.scheduler.step()
            
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            # LR Decay - 移到这里
            self.scheduler.step()

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            # if epoch > 1:
            #     self.logger.info("Saving log_image")
            #     image_prefix = '{}/latest'.format(self.result_folder)
            #     util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
            #                         self.result_log, labels=['train_score'])
            #     util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
            #                         self.result_log, labels=['train_loss'])

            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                try:
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'], 
                                                   self.result_log, labels=['train_score'])
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'], 
                                                   self.result_log, labels=['train_loss'])
                except Exception as e:
                    self.logger.warning(f"Failed to save log image: {e}")

            
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # if all_done or (epoch % img_save_interval) == 0:
            #     image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
            #     util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
            #                         self.result_log, labels=['train_score'])
            #     util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
            #                         self.result_log, labels=['train_loss'])

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                try:
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                                   self.result_log, labels=['train_score'])
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                   self.result_log, labels=['train_loss'])
                except Exception as e:
                    self.logger.warning(f"Failed to save log image: {e}")

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        
        tw_violation_AM = AverageMeter() if self.use_time_window_constraint else None
        energy_violation_AM = AverageMeter() if self.use_energy_constraint else None

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss, constraint_stats = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            
            if constraint_stats is not None:
                if self.use_time_window_constraint and 'tw_violation' in constraint_stats:
                    tw_violation_AM.update(constraint_stats['tw_violation'], batch_size)
                if self.use_energy_constraint and 'energy_violation' in constraint_stats:
                    energy_violation_AM.update(constraint_stats['energy_violation'], batch_size)

            episode += batch_size

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    msg = 'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'.format(
                        epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                        score_AM.avg, loss_AM.avg)
                    if self.use_time_window_constraint or self.use_energy_constraint:
                        if self.use_time_window_constraint:
                            msg += f',  TW_Vio: {tw_violation_AM.avg:.4f}'
                        if self.use_energy_constraint:
                            msg += f',  E_Vio: {energy_violation_AM.avg:.4f}'
                    self.logger.info(msg)

        msg = 'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'.format(
            epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg)
        if self.use_time_window_constraint or self.use_energy_constraint:
            if self.use_time_window_constraint:
                msg += f',  TW_Vio: {tw_violation_AM.avg:.4f}'
            if self.use_energy_constraint:
                msg += f',  E_Vio: {energy_violation_AM.avg:.4f}'
        self.logger.info(msg)

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))

        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        log_prob = prob_list.log().sum(dim=2)
        loss = -advantage * log_prob
        loss_mean = loss.mean()

        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()

        constraint_stats = None
        if self.use_time_window_constraint or self.use_energy_constraint:
            constraint_stats = {}
            if self.use_time_window_constraint:
                tw_violations = state.time_window_violation.float().mean().item()
                constraint_stats['tw_violation'] = tw_violations
            if self.use_energy_constraint:
                energy_violations = state.energy_shortage.float().mean().item()
                constraint_stats['energy_violation'] = energy_violations

        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item(), constraint_stats