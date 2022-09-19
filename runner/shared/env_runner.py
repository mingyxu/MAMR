import numpy as np
from numpy.core.fromnumeric import argmax
import torch
from runner.shared.base_runner import Runner
import os
import random
import time
import pandas as pd
from pathlib import Path

def _t2n(x):
    return x.detach().cpu().numpy()

hex_weight = pd.read_csv("./data/hex_bins/hex_weight.csv").values


class EnvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        _step_start = 0
        best_reject = 1
        
        for episode in range(episodes):
            model = self.trainer.policy
            start = time.time()
            self.warmup(_step_start)
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            for step in range(self.episode_length):
                # buffer: 108*6000
                # Sample actions
                values, st_values, actions, action_log_probs, actions_env = self.collect(step)

                # Obser reward and next obs
                results = self.envs.step(actions_env, model)
                try:
                    obs, rewards, dones, infos, cent_state = (np.stack(result) for result in zip(*results))
                except:
                    obs, rewards, dones, infos, cent_state, reject_rate, total_reward, trv_time, income = (np.stack(result) for result in zip(*results))
                data = obs, rewards, dones, infos, values, st_values, actions, action_log_probs, cent_state

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, time {}."
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              (end - start)/60), flush=True)
               
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["reject_rate"] = reject_rate[0]
                train_infos["total_reward"] = total_reward[0]
                print(" average episode rewards is {}\n".format(train_infos["average_episode_rewards"]), flush=True)
                self.log_train(train_infos, episode)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                eval_reject_rate, eval_total_reward, eval_trv_time, eval_income = self.eval(episode)
                # save model
                if eval_reject_rate < best_reject:
                    self.save()
                    best_reject = eval_reject_rate
                    with open(os.path.join(self.txt_dir, 'best.txt'), 'w') as f:
                        f.write("episode: "+str(episode)+" "+"rej: "+str(best_reject)+" "+"rew: " +str(eval_total_reward)+" "+"time: "+str(eval_trv_time)+" "+str(1-eval_trv_time/(self.num_agents*108))+" "+"income: "+str(eval_income))

            if _step_start % 3204 == 0 and _step_start != 0:
                _step_start = 36
            else:
                _step_start += 144

    def warmup(self, _step_start):
        # reset env
        model = self.trainer.policy

        obs, infos, cent_state = self.envs.reset(_step_start, model)  

        if self.use_centralized_V:  
            share_obs = cent_state.reshape(self.n_rollout_threads, -1)  
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)  
            hexbin = infos[0,:,1].reshape(1, self.num_agents, 1) 
            share_obs = np.concatenate((hexbin, share_obs), axis=2) 

            weights = hex_weight[infos[0,:,1]] 
            temp = share_obs[0,:,3:] * weights
            share_obs[0,:,3:] = temp
        else:
            share_obs = obs

        _available_actions = []
        for i in range(self.num_agents):
            if infos[0][i][0] == 0:
                _available_actions.append([1,1,1,1,1,1,1,0])
            else:
                _available_actions.append([0,0,0,0,0,0,0,1])
        
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = _available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, st_values, action, action_log_prob \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.masks[step]),
                                              np.concatenate(self.buffer.available_actions[step]),
                                              deterministic=False)
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        st_values = np.array(np.split(_t2n(st_values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, st_values, actions, action_log_probs, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, st_values, actions, action_log_probs, cent_state = data

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.int8)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.int8)

        if self.use_centralized_V:
            share_obs = cent_state.reshape(self.n_rollout_threads, -1)  
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)  
            hexbin = infos[0,:,1].reshape(1, self.num_agents, 1)
            share_obs = np.concatenate((hexbin, share_obs), axis=2)  

            weights = hex_weight[infos[0,:,1]]
            temp = share_obs[0,:,3:] * weights
            share_obs[0,:,3:] = temp
        else:
            share_obs = obs
        _available_actions = []
        for i in range(self.num_agents):
            if infos[0][i][0] == 0:
                _available_actions.append([1,1,1,1,1,1,1,0])
            else:
                _available_actions.append([0,0,0,0,0,0,0,1])
        self.buffer.insert(share_obs, obs, actions, action_log_probs, values, st_values, rewards,
                           masks, available_actions=_available_actions)

    @torch.no_grad()
    def eval(self, episode):
        eval_episode_rewards = []
        model = self.trainer.policy

        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.int8)
        
        _step_start_test = 36
        eval_reject_rates = []
        eval_total_rewards = []
        eval_trv_times = []
        eval_incomes = []
        for i in range(7): # 7
            eval_obs, eval_infos, eval_cent_state = self.eval_envs.reset(_step_start_test, model)
           
            eval_available_actions = []
            for i in range(self.num_agents):
                if eval_infos[0][i][0] == 0:
                    eval_available_actions.append([1,1,1,1,1,1,1,0])
                else:
                    eval_available_actions.append([0,0,0,0,0,0,0,1])

            for test_step in range(self.episode_length):
                self.trainer.prep_rollout()

                eval_action = self.trainer.policy.act(np.concatenate(eval_obs),
                                                        np.concatenate(eval_masks),
                                                        available_actions=np.array(eval_available_actions),
                                                        deterministic=False)

                eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads)) 
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)

                # Obser reward and next obs
                results = self.eval_envs.step(eval_actions_env, model)

                try:
                    eval_obs, eval_rewards, eval_dones, eval_infos, eval_cent_state = (np.stack(result) for result in zip(*results))
                except:
                    eval_obs, eval_rewards, eval_dones, eval_infos, eval_cent_state, eval_reject_rate, eval_total_reward, eval_trv_time, eval_income = (np.stack(result) for result in zip(*results))
                    eval_reject_rates.append(eval_reject_rate[0])
                    eval_total_rewards.append(eval_total_reward[0])
                    eval_trv_times.append(eval_trv_time[0])
                    eval_incomes.append(eval_income[0])
                
                eval_episode_rewards.append(eval_rewards) 

                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.int8)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.int8)

                eval_available_actions = []
                for i in range(self.num_agents):
                    if eval_infos[0][i][0] == 0:
                        eval_available_actions.append([1,1,1,1,1,1,1,0])
                    else:
                        eval_available_actions.append([0,0,0,0,0,0,0,1])


            if _step_start_test % 900 == 0 and _step_start_test != 0:
                _step_start_test = 36
            else:
                _step_start_test += 144

        eval_env_infos = {}
        eval_env_infos['eval_reject_rate'] = np.mean(eval_reject_rates)
        eval_env_infos['eval_total_reward'] = np.mean(eval_total_rewards)
        print("eval reject rate: ", eval_env_infos['eval_reject_rate'], flush=True)
        print("eval total income: ", np.mean(eval_incomes), flush=True)
        print("eval idle time: ", (1-np.mean(eval_trv_times)/(self.num_agents*108)), flush=True)
        
        self.log_env(eval_env_infos, episode)

        return eval_env_infos['eval_reject_rate'], eval_env_infos['eval_total_reward'], np.mean(eval_trv_times), np.mean(eval_incomes)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        pass