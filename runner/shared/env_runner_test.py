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


class EnvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):
        best_reject = 1        
        # test
        eval_reject_rate, eval_total_reward, eval_trv_time, eval_income = self.eval(1)               
        best_reject = eval_reject_rate
        txt_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results")
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        with open(os.path.join(txt_dir, 'best.txt'), 'w') as f:
            f.write("rejection rate: "+str(best_reject)+" "+"reward: " +str(eval_total_reward)+" "+"time: "+str(eval_trv_time)+" "+str(1-eval_trv_time/(self.num_agents*108))+" "+"income: "+str(eval_income))
    
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
        for i in range(1): #TODO 7
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