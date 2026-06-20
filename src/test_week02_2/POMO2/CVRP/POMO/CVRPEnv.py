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
    tw_early: torch.Tensor = None
    tw_late: torch.Tensor = None
    initial_battery: torch.Tensor = None
    ev_params: dict = None


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    selected_count: int = None
    load: torch.Tensor = None
    current_time: torch.Tensor = None
    current_battery: torch.Tensor = None
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    finished: torch.Tensor = None
    time_window_violation: torch.Tensor = None
    energy_shortage: torch.Tensor = None


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
        
        self.node_service_time = None
        self.tw_early = None
        self.tw_late = None
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
        
        self.time_window_violation = None
        self.energy_shortage = None

        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
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
                    self.node_service_time = problem_data['node_service_time']
                    self.tw_early = problem_data['tw_early']
                    self.tw_late = problem_data['tw_late']
                
                if self.use_energy_constraint:
                    self.initial_battery = problem_data['initial_battery']
            else:
                depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.problem_size)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                
                if self.use_time_window_constraint:
                    self.node_service_time = self.node_service_time.repeat(8, 1)
                    self.tw_early = self.tw_early.repeat(8, 1)
                    self.tw_late = self.tw_late.repeat(8, 1)
                
                if self.use_energy_constraint:
                    self.initial_battery = self.initial_battery.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        
        if self.use_time_window_constraint:
            self.reset_state.node_service_time = self.node_service_time
            self.reset_state.tw_early = self.tw_early
            self.reset_state.tw_late = self.tw_late
        
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
            self.current_battery = self.initial_battery.expand(self.batch_size, self.pomo_size).clone()
        else:
            self.current_battery = torch.zeros(size=(self.batch_size, self.pomo_size))
        
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        
        self.time_window_violation = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        self.energy_shortage = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)

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
        self.step_state.time_window_violation = self.time_window_violation
        self.step_state.energy_shortage = self.energy_shortage

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        self.at_the_depot = (selected == 0)

        if self.selected_count > 1:
            prev_node = self.selected_node_list[:, :, -2]
            prev_xy = self.depot_node_xy[self.BATCH_IDX, prev_node, :]
            curr_xy = self.depot_node_xy[self.BATCH_IDX, selected, :]
            
            distance = torch.sqrt(((prev_xy - curr_xy) ** 2).sum(dim=2) + 1e-8)
            
            if self.use_time_window_constraint:
                travel_time = distance / 1.0
                arrival_time = self.current_time + travel_time
                
                service_time = torch.zeros_like(distance)
                tw_early = torch.zeros_like(distance)
                tw_late = torch.ones_like(distance)
                
                for b in range(self.batch_size):
                    for p in range(self.pomo_size):
                        node_idx = selected[b, p].item()
                        if node_idx > 0:
                            service_time[b, p] = self.node_service_time[b, node_idx - 1]
                            tw_early[b, p] = self.tw_early[b, node_idx - 1]
                            tw_late[b, p] = self.tw_late[b, node_idx - 1]
                
                too_early = arrival_time < tw_early
                waiting_time = torch.where(too_early, tw_early - arrival_time, torch.zeros_like(arrival_time))
                actual_arrival = torch.maximum(arrival_time, tw_early)
                
                too_late = actual_arrival > tw_late
                self.time_window_violation = too_late
                
                self.current_time = actual_arrival + service_time
            
            if self.use_energy_constraint:
                energy_consumed = distance * self.ev_params['energy_consumption_rate']
                self.current_battery -= energy_consumed
                
                if self.at_the_depot.any():
                    self.current_battery[self.at_the_depot] = self.ev_params['battery_capacity']
                
                self.energy_shortage = self.current_battery < 0
                self.current_battery = torch.clamp(self.current_battery, min=0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        gathering_index = selected[:, :, None]
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0

        self.ninf_mask = self.visited_ninf_flag.clone()
        
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        self.ninf_mask[demand_too_large] = float('-inf')
        
        if self.use_energy_constraint:
            for b in range(self.batch_size):
                for p in range(self.pomo_size):
                    if self.current_battery[b, p] > 0:
                        curr_node = self.current_node[b, p].item()
                        curr_xy = self.depot_node_xy[b, curr_node, :]
                        for node_idx in range(self.problem_size + 1):
                            if self.ninf_mask[b, p, node_idx] != float('-inf'):
                                node_xy = self.depot_node_xy[b, node_idx, :]
                                dist_to_node = torch.sqrt(((curr_xy - node_xy) ** 2).sum() + 1e-8)
                                energy_needed = dist_to_node * self.ev_params['energy_consumption_rate']
                                min_energy_needed = energy_needed + dist_to_node * 0.5
                                if self.current_battery[b, p] < min_energy_needed:
                                    self.ninf_mask[b, p, node_idx] = float('-inf')

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        self.finished = self.finished + newly_finished

        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_time = self.current_time
        self.step_state.current_battery = self.current_battery
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.time_window_violation = self.time_window_violation
        self.step_state.energy_shortage = self.energy_shortage

        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()
            
            if self.use_time_window_constraint:
                tw_penalty = self.time_window_violation.sum() * 100
                reward = reward - tw_penalty
            
            if self.use_energy_constraint:
                energy_penalty = self.energy_shortage.sum() * 100
                reward = reward - energy_penalty
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()

        travel_distances = segment_lengths.sum(2)
        return travel_distances