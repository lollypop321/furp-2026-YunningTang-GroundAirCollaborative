from dataclasses import dataclass
import torch

from CVRProblemDef import get_random_problems, get_random_problems_with_ev_tw, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    node_service_time: torch.Tensor = None
    # shape: (batch, problem)
    tw_early: torch.Tensor = None
    # shape: (batch, problem)
    tw_late: torch.Tensor = None
    # shape: (batch, problem)
    initial_battery: torch.Tensor = None
    # shape: (batch, 1)
    ev_params: dict = None


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_time: torch.Tensor = None
    # shape: (batch, pomo)
    current_battery: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class CVRPEnv:
    def __init__(self, **env_params):

        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        self.use_energy_constraint = env_params.get('use_energy_constraint', False)
        self.use_time_window_constraint = env_params.get('use_time_window_constraint', False)
        self.ev_params = env_params.get('ev_params', {
            'battery_capacity': 100.0,
            'energy_consumption_rate': 0.5,
            'charging_speed': 1.0,
        })

        # 惩罚系数：使用 soft penalty with annealing
        self.tw_penalty_coeff = env_params.get('tw_penalty_coeff', 1.0)
        self.energy_penalty_coeff = env_params.get('energy_penalty_coeff', 1.0)

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.depot_node_xy = None
        self.depot_node_demand = None

        # TW 相关（depot 前缀版，含 depot 位置，方便 gather）
        self.depot_node_service_time = None  # shape: (batch, problem+1)
        self.depot_node_tw_early = None      # shape: (batch, problem+1)
        self.depot_node_tw_late = None       # shape: (batch, problem+1)
        self.initial_battery = None

        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None

        self.at_the_depot = None
        self.load = None
        self.current_time = None
        self.current_battery = None
        self.visited_ninf_flag = None
        self.ninf_mask = None
        self.finished = None

        # 累积违反计数
        self.total_tw_violation = None       # shape: (batch, pomo) - 累积 TW 违反次数
        self.total_tw_delay = None           # shape: (batch, pomo) - 累积 TW 延迟量
        self.total_energy_shortage = None    # shape: (batch, pomo) - 累积能量不足次数

        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        
        # 加载 TW/EV 相关字段（如果存在）
        if self.use_time_window_constraint:
            self.saved_node_service_time = loaded_dict.get('node_service_time', None)
            self.saved_tw_early = loaded_dict.get('tw_early', None)
            self.saved_tw_late = loaded_dict.get('tw_late', None)
        
        if self.use_energy_constraint:
            self.saved_initial_battery = loaded_dict.get('initial_battery', None)
        
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            if self.use_energy_constraint or self.use_time_window_constraint:
                problem_data = get_random_problems_with_ev_tw(
                    batch_size,
                    self.problem_size,
                    use_ev=self.use_energy_constraint,
                    use_tw=self.use_time_window_constraint
                )
                depot_xy = problem_data['depot_xy']
                node_xy = problem_data['node_xy']
                node_demand = problem_data['node_demand']

                if self.use_time_window_constraint:
                    node_service_time = problem_data['node_service_time']
                    tw_early = problem_data['tw_early']
                    tw_late = problem_data['tw_late']

                if self.use_energy_constraint:
                    self.initial_battery = problem_data['initial_battery']
            else:
                depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.problem_size)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index + batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index + batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index + batch_size]
            
            # 加载 TW/EV 相关字段
            if self.use_time_window_constraint:
                node_service_time = self.saved_node_service_time[self.saved_index:self.saved_index + batch_size]
                tw_early = self.saved_tw_early[self.saved_index:self.saved_index + batch_size]
                tw_late = self.saved_tw_late[self.saved_index:self.saved_index + batch_size]
            
            if self.use_energy_constraint:
                self.initial_battery = self.saved_initial_battery[self.saved_index:self.saved_index + batch_size]
            
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)

                if self.use_time_window_constraint:
                    node_service_time = node_service_time.repeat(8, 1)
                    tw_early = tw_early.repeat(8, 1)
                    tw_late = tw_late.repeat(8, 1)

                if self.use_energy_constraint:
                    self.initial_battery = self.initial_battery.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        # 构建含 depot 的 TW 数据（depot 索引为 0，时间窗设为 [0, +inf]）
        if self.use_time_window_constraint:
            depot_service_time = torch.zeros(size=(self.batch_size, 1))
            self.depot_node_service_time = torch.cat((depot_service_time, node_service_time), dim=1)
            # shape: (batch, problem+1)

            depot_tw_early = torch.zeros(size=(self.batch_size, 1))
            self.depot_node_tw_early = torch.cat((depot_tw_early, tw_early), dim=1)
            # shape: (batch, problem+1)

            depot_tw_late = torch.ones(size=(self.batch_size, 1)) * 1e6  # depot 无时间窗约束
            self.depot_node_tw_late = torch.cat((depot_tw_late, tw_late), dim=1)
            # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        if self.use_time_window_constraint:
            self.reset_state.node_service_time = node_service_time
            self.reset_state.tw_early = tw_early
            self.reset_state.tw_late = tw_late

        if self.use_energy_constraint:
            self.reset_state.initial_battery = self.initial_battery
            self.reset_state.ev_params = self.ev_params

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))

        self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size))
        if self.use_energy_constraint:
            self.current_battery = torch.ones(size=(self.batch_size, self.pomo_size)) * self.ev_params['battery_capacity']
        else:
            self.current_battery = torch.zeros(size=(self.batch_size, self.pomo_size))

        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)

        # 累积违反计数
        self.total_tw_violation = torch.zeros(size=(self.batch_size, self.pomo_size))
        self.total_tw_delay = torch.zeros(size=(self.batch_size, self.pomo_size))
        self.total_energy_shortage = torch.zeros(size=(self.batch_size, self.pomo_size))

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_time = self.current_time
        self.step_state.current_battery = self.current_battery
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        self.at_the_depot = (selected == 0)

        # ====== 时间 & 能量更新（向量化） ======
        if self.selected_count > 1:
            prev_node = self.selected_node_list[:, :, -2]
            # shape: (batch, pomo)

            # 获取前一个节点和当前节点的坐标
            prev_xy = self.depot_node_xy[self.BATCH_IDX, prev_node, :]
            curr_xy = self.depot_node_xy[self.BATCH_IDX, selected, :]
            # shape: (batch, pomo, 2)

            distance = torch.sqrt(((prev_xy - curr_xy) ** 2).sum(dim=2) + 1e-8)
            # shape: (batch, pomo)

            # --- 时间窗约束（向量化，无 for 循环） ---
            if self.use_time_window_constraint:
                travel_time = distance  # 假设速度为 1

                # 用 gather 获取当前选中节点的 service_time, tw_early, tw_late
                gather_idx = selected  # shape: (batch, pomo)
                service_time = self.depot_node_service_time.unsqueeze(1).expand(
                    -1, self.pomo_size, -1).gather(dim=2, index=gather_idx.unsqueeze(2)).squeeze(2)
                tw_early = self.depot_node_tw_early.unsqueeze(1).expand(
                    -1, self.pomo_size, -1).gather(dim=2, index=gather_idx.unsqueeze(2)).squeeze(2)
                tw_late = self.depot_node_tw_late.unsqueeze(1).expand(
                    -1, self.pomo_size, -1).gather(dim=2, index=gather_idx.unsqueeze(2)).squeeze(2)
                # shape: (batch, pomo)

                arrival_time = self.current_time + travel_time

                # 如果到达太早，等待到 tw_early
                actual_start = torch.maximum(arrival_time, tw_early)

                # 检查是否违反 tw_late
                delay = (actual_start - tw_late).clamp(min=0.0)
                too_late = delay > 0
                # shape: (batch, pomo)

                # 累积违反
                self.total_tw_violation += too_late.float()
                self.total_tw_delay += delay

                # 更新当前时间
                self.current_time = actual_start + service_time

                # depot 重置时间
                self.current_time[self.at_the_depot] = 0.0

            # --- 能量约束（向量化） ---
            if self.use_energy_constraint:
                energy_consumed = distance * self.ev_params['energy_consumption_rate']
                self.current_battery -= energy_consumed

                # 检查能量不足
                shortage = (self.current_battery < 0).float()
                self.total_energy_shortage += shortage

                # clamp 到 0（允许继续但记录违反）
                self.current_battery = torch.clamp(self.current_battery, min=0)

                # depot 充满电
                self.current_battery[self.at_the_depot] = self.ev_params['battery_capacity']

        # ====== 容量约束 ======
        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # depot 补满

        # ====== 更新访问标记 & mask ======
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot 始终可访问（除非正在 depot）

        self.ninf_mask = self.visited_ninf_flag.clone()

        # 容量 mask
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')

        # 能量 mask（向量化：禁止无法到达的节点）
        if self.use_energy_constraint:
            # 当前节点坐标
            curr_xy = self.depot_node_xy[self.BATCH_IDX, self.current_node, :]
            # shape: (batch, pomo, 2)

            # 所有节点坐标
            all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
            # shape: (batch, pomo, problem+1, 2)

            # 到所有节点的距离
            dist_to_all = torch.sqrt(((curr_xy[:, :, None, :] - all_xy) ** 2).sum(dim=3) + 1e-8)
            # shape: (batch, pomo, problem+1)

            # depot 坐标
            depot_xy = self.depot_node_xy[:, 0:1, :].expand(-1, self.problem_size + 1, -1)
            # shape: (batch, problem+1, 2)
            # 每个节点到 depot 的距离
            dist_node_to_depot = torch.sqrt(((self.depot_node_xy - depot_xy) ** 2).sum(dim=2) + 1e-8)
            # shape: (batch, problem+1)
            dist_node_to_depot = dist_node_to_depot[:, None, :].expand(-1, self.pomo_size, -1)
            # shape: (batch, pomo, problem+1)

            # 需要的总能量 = 到目标节点 + 从目标节点回 depot（保守估计）
            energy_to_node = dist_to_all * self.ev_params['energy_consumption_rate']
            energy_node_to_depot = dist_node_to_depot * self.ev_params['energy_consumption_rate']
            total_energy_needed = energy_to_node + energy_node_to_depot

            # 当前电量不足以到达并返回 depot 的节点被 mask
            insufficient_energy = self.current_battery[:, :, None] < total_energy_needed
            # shape: (batch, pomo, problem+1)

            # 只 mask 尚未被 mask 的节点
            self.ninf_mask[insufficient_energy] = float('-inf')

            # depot 永远不被能量 mask（可以回去充电）
            self.ninf_mask[:, :, 0][self.ninf_mask[:, :, 0] == float('-inf')] = 0
            # 恢复 depot 的 mask 状态应遵循 visited_ninf_flag
            depot_should_be_masked = self.visited_ninf_flag[:, :, 0] == float('-inf')
            self.ninf_mask[:, :, 0][depot_should_be_masked] = float('-inf')

        # 时间窗 mask（可选：禁止已确定无法在 tw_late 前到达的节点）
        if self.use_time_window_constraint:
            curr_xy = self.depot_node_xy[self.BATCH_IDX, self.current_node, :]
            # shape: (batch, pomo, 2)
            all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
            # shape: (batch, pomo, problem+1, 2)
            dist_to_all = torch.sqrt(((curr_xy[:, :, None, :] - all_xy) ** 2).sum(dim=3) + 1e-8)
            # shape: (batch, pomo, problem+1)

            earliest_arrival = self.current_time[:, :, None] + dist_to_all
            # shape: (batch, pomo, problem+1)

            tw_late_all = self.depot_node_tw_late[:, None, :].expand(-1, self.pomo_size, -1)
            # shape: (batch, pomo, problem+1)

            # 如果最早到达时间已经超过 tw_late，该节点不可行
            tw_infeasible = earliest_arrival > tw_late_all
            self.ninf_mask[tw_infeasible] = float('-inf')

            # 恢复 depot 状态
            self.ninf_mask[:, :, 0] = self.visited_ninf_flag[:, :, 0].clone()
            # depot 对于未 finished 的不在 depot 的 agent 应该可用
            self.ninf_mask[:, :, 0][~self.at_the_depot & ~self.finished] = 0

        # ====== 结束判断 ======
        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished | newly_finished

        # finished 的 episode depot 保持可用（让它们停在 depot）
        self.ninf_mask[:, :, 0][self.finished] = 0

        # ====== 更新 step_state ======
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_time = self.current_time
        self.step_state.current_battery = self.current_battery
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # ====== 计算 reward ======
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()

            # Soft penalty（基于累积违反量）
            if self.use_time_window_constraint:
                # 惩罚 = 系数 × 平均延迟时间
                tw_penalty = self.tw_penalty_coeff * self.total_tw_delay
                reward = reward - tw_penalty

            if self.use_energy_constraint:
                energy_penalty = self.energy_penalty_coeff * self.total_energy_shortage
                reward = reward - energy_penalty
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances
