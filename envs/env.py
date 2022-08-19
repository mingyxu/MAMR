from os import error
import numpy as np
from data.data_provider import DataProvider
from envs.simulator import initialize_driver_distribution
from envs.simulator import take_action
from envs.simulator import take_dispatch_action
import torch
import os
from pathlib import Path


""" for one taxi """

class Env(object):
    def __init__(self, i):
        self.agent_num = 9000 
        self.obs_dim = 10  
        self.action_dim = 8  # 0~5：neighbors 6：wait 7：order

        # City state
        DATA_DIR = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/data")
        self.data_provider = DataProvider(DATA_DIR)
        self.hex_attr_df = self.data_provider.read_hex_bin_attributes()
        self.hex_bins = self.hex_attr_df['hex_id']
        self.hex_nei = self.hex_attr_df[['hex_id', 'north_east_neighbor', 'north_neighbor',
                                         'north_west_neighbor', 'south_east_neighbor',
                                         'south_neighbor', 'south_west_neighbor']]

        self.T = 108  #144  # Number of time steps
        self.S = len(self.hex_bins)  # Number of hex bins
        self.step_start = 0
        self.num_drivers = self.agent_num

    def reset(self, step_start=36, mode="train", model=None):
        # dataset
        if mode == "test":
            print("testing", flush=True)
            self.city_states = self.data_provider.read_city_states(filename='city_states_test.dill')
        else:
            print("\ntraining", flush=True)
            self.city_states = self.data_provider.read_city_states(filename='city_states_train.dill')
        # initialize env_supply
        self.t = 0
        self.total_reward = 0
        self.total_trv_time = 0
        self.total_income = 0
        self.reject_rate = 0
        self.unfulfilled_demand = 0
        self.all_demand = 0
        self.step_start = step_start
        self.curr_driver_distribution = initialize_driver_distribution(self.num_drivers)
        self.driver_distribution_matrix = np.empty((self.T, self.S), dtype=object)
        self.driver_distribution_matrix.fill([])
        self.driver_distribution_matrix[0] = self.curr_driver_distribution
        env_state_supply = self._get_obs()
        # 0. idle ;  1. c grid ; 2. d time ; 3. d grid ; 4. idle_duration
        self.driver_state = np.zeros((self.agent_num,4), dtype=int)
        # first dispatch
        timestamp = self.city_states[step_start]['time']
        day = timestamp.date()
        self.day = timestamp.day
        self.dayofweek = day.weekday()
        self.dispatch(model)
        # get initial states
        sub_agent_obs = []
        sub_agent_info = []
        for i in range(self.agent_num):
            h = self.driver_state[i][1]
            neighbors = self.get_neighbors(h)
            sub_obs = self.get_state(h, neighbors, env_state_supply, self.city_state_demand, self.t)
            sub_agent_obs.append(sub_obs)
            sub_agent_info.append([self.driver_state[i][0], self.driver_state[i][1]])
        cent_state = self.get_city_SDGap(env_state_supply, self.city_state_demand, self.t)
        sub_agent_info = np.array(sub_agent_info)
        return [sub_agent_obs, sub_agent_info, cent_state]

    def step(self, actions, model=None):
        # dispatch at the beginning of each time step
        if self.t != 0:
            self.dispatch(model)
        # next demand for calculating the balance degree
        if self.t < (self.T-2):
            next_city_demand = self.city_states[self.t + self.step_start + 1]['pickup_vector']
        else:
            next_city_demand = np.zeros((142,))

        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []

        for i in range(self.agent_num):
            # take action
            if self.driver_state[i][0] == 0: # idle car >>> relocation
                driver_distribution_matrix, reward, city_state_demand, driver_state_i, hex_demand_all, trv_time, income = take_action(self.t, actions[i],
                    self.city_states[self.t + self.step_start], self.hex_nei, self.driver_distribution_matrix.copy(),
                    self.T, i, self.driver_state[i][1], self.city_state_demand.copy(), next_city_demand, self.driver_state[i].copy(), self.hex_demand.copy())
                self.total_reward += reward
                self.total_income += income
                self.total_trv_time += trv_time
                self.driver_distribution_matrix = driver_distribution_matrix
                self.city_state_demand = city_state_demand
                self.driver_state[i] = driver_state_i
                self.hex_demand = hex_demand_all
                sub_agent_reward.append(reward)
            else: 
                sub_agent_reward.append(0)

        # current demand: unfulfilled demand
        self.unfulfilled_demand += np.sum(self.city_state_demand)
        self.all_demand += np.sum(self.city_states[self.t + self.step_start]['pickup_vector'])
        if np.sum(self.city_state_demand) != np.sum(self.hex_demand):
            raise RuntimeError
        # Update time step, next time step
        if self.t == (self.T-1):
            self.curr_driver_distribution = self.driver_distribution_matrix[self.T - 1]
            self.city_state_demand = self.city_states[self.T - 1 + self.step_start]['pickup_vector']
            done = True
            self.reject_rate = self.unfulfilled_demand / self.all_demand
        else:
            self.t += 1
            self.curr_driver_distribution = self.driver_distribution_matrix[self.t]
            self.city_state_demand = self.city_states[self.t + self.step_start]['pickup_vector']
            done = False
        # next state
        env_state_supply = self._get_obs()
        for i in range(self.agent_num):
            # if arrivied at the drop-off grid
            if self.driver_state[i][0] == 1 and self.driver_state[i][2] == self.t: 
                self.driver_state[i][1] = self.driver_state[i][3]
                self.driver_state[i][0] = 0
            h = self.driver_state[i][1]
            neighbors = self.get_neighbors(h)
            sub_obs = self.get_state(h, neighbors, env_state_supply, self.city_state_demand, self.t)
            sub_agent_obs.append(sub_obs)
            sub_agent_info.append([self.driver_state[i][0], self.driver_state[i][1]])
            sub_agent_done.append(done)
        cent_state = self.get_city_SDGap(env_state_supply, self.city_state_demand, self.t)
        
        sub_agent_info = np.array(sub_agent_info)
        if done:
            print("reject_rate:", self.reject_rate, flush=True)
            print("unfulfilled_demand", self.unfulfilled_demand, flush=True)
            print("all_demand", self.all_demand, flush=True)
            return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info, cent_state, self.reject_rate, self.total_reward, self.total_trv_time, self.total_income]
        else:
            return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info, cent_state]
    
    def get_state(self, h, neighbors, env_state, city_state, step):
        """
        env: supply > drivers
        city:demand > orders
         """
        # hexbinID + timestep + dayofweek + supply/demand(gap) current/neighbor
        gap = [] # S-D
        for i in range(len(neighbors)):
            if neighbors[i]==' ':
                gap.append(0)
            else:
                neighbor = int(neighbors[i])
                gap.append(env_state[neighbor]-city_state[neighbor])
        state = []
        step = step % 144
        state.append(h) # current hexbinID 2                     
        state.append(step)  # current timestep  0   
        state.append(self.dayofweek)  # dayofweek  1  
        state.append(env_state[h]-city_state[h])   # gap 3
        for j in range(len(neighbors)):   # gap
            state.append(gap[j])
        state = np.array(state) 
        return state

    def get_city_SDGap(self, env_state, city_state, step):
        city_SDGap = []
        city_SDGap.append(step)
        city_SDGap.append(self.dayofweek)
        city_SDGap.extend(env_state-city_state)
        return city_SDGap

    def get_neighbors(self, h):
        neighbors = self.hex_nei.iloc[h].values[1:]
        return neighbors

    def _get_obs(self, curr_driver_distribution=None):
        """
        Returns an observation
        :param curr_driver_distribution:
        :return obs:
        """
        if curr_driver_distribution is not None:
            obs = np.array([len(x) for x in curr_driver_distribution])
        else:
            obs = np.array([len(x) for x in self.curr_driver_distribution])
        return obs

    def dispatch(self, model=None):
        rewards = 0
        city_state_demand = []
        self.hex_demand = self.city_states[self.t + self.step_start]['ride_count_matrix'].astype(int)
        for hex_bin in range(self.S):
            hex_supply = self._get_obs()[hex_bin] 
            if hex_supply >= np.sum(self.hex_demand[hex_bin]) or hex_supply==0 or np.sum(self.hex_demand[hex_bin])==0: # S >= D
                index = None
            else:
                index = self.get_deindex(hex_bin, model) # if S < D index else none
            drivers = list(self.driver_distribution_matrix[self.t][hex_bin])
            driver_distribution_matrix, reward, hex_state_demand, driver_state, hex_demand, trv_time = take_dispatch_action(self.t, self.T, self.S, drivers, hex_bin, self.city_states[self.t + self.step_start], self.driver_distribution_matrix.copy(), self.driver_state.copy(), index)
            rewards += reward
            self.total_trv_time += trv_time
            city_state_demand.append(hex_state_demand)
            self.driver_distribution_matrix = driver_distribution_matrix
            self.driver_state = driver_state
            self.hex_demand[hex_bin] = hex_demand
        
        # current demand, save it in the simulator
        self.city_state_demand = city_state_demand
        if np.sum(self.city_state_demand) != np.sum(self.hex_demand):
            raise RuntimeError

        # Return current supply (if the order-driver match is successful, 
        # the vehicle will appear in a future area)
        if self.t == (self.T-1):
            self.curr_driver_distribution = self.driver_distribution_matrix[self.T - 2]
        else:
            self.curr_driver_distribution = self.driver_distribution_matrix[self.t]

        self.total_reward += rewards
        self.total_income += rewards
    
    @torch.no_grad()         
    def get_deindex(self, hex_bin, model):
        """ 
        param: hexbinID + timestep + dayofweek
        function: calculate advantage of each order
        return: index of descending weight
        """
        advantage = []
        gamma = 0.92
        count = self.city_states[self.t + self.step_start]['ride_count_matrix'][hex_bin].astype(int)
        reward = self.city_states[self.t + self.step_start]['distance_matrix'][hex_bin]*1.5
        time = self.city_states[self.t + self.step_start]['travel_time_matrix'][hex_bin].astype(int)
        s_t = [hex_bin, self.t, self.dayofweek]
        s_t = np.array(s_t).reshape(1,-1)
        v_t = model.get_st_values(torch.Tensor(s_t))
        for i in range(self.S):
            if count[i] > 0:
                h_i = i
                delta_t = (time[i] if time[i]>1 else 1)
                t_i = self.t + delta_t
                s_nt = [h_i, t_i, self.dayofweek]
                s_nt = np.array(s_nt).reshape(1,-1)
                v_nt = model.get_st_values(torch.Tensor(s_nt))
                ad = v_t - (gamma**delta_t) * v_nt - reward[i]
                advantage.append(ad.cpu().detach())
            else:
                advantage.append(100)
        de_index = np.argsort(advantage)
        return de_index
