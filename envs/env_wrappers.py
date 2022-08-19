"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""

import numpy as np
import gym
from gym import spaces
from envs.env import Env


class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    def __init__(self, array_of_param_array):
        super().__init__()
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (
                    np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


class SubprocVecEnv(object):
    def __init__(self, all_args):
        """
        envs: list of gym environments to run in subprocesses
        for train
        """

        self.env_list = [Env(i) for i in range(all_args.n_rollout_threads)]
        self.num_envs = all_args.n_rollout_threads

        self.num_agent = self.env_list[0].agent_num
        self.signal_obs_dim = self.env_list[0].obs_dim
        self.signal_action_dim = self.env_list[0].action_dim

        self.u_range = 1.0  # control range for continuous control
        self.movable = True

        # environment parameters
        # self.discrete_action_space = True
        self.discrete_action_space = True

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = False

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 145 
        st_obs_dim = 3
        for agent in range(self.num_agent):
            total_action_space = []

            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(self.signal_action_dim) 
            else:
                u_action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(6000,), dtype=np.int8)  # [-1,1]
            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            # share_obs_dim += self.signal_obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,),
                                                     dtype=np.float32))  # [-inf,inf]

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.num_agent)]
        self.st_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(st_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.num_agent)]

    def step(self, actions, _model):
        results = [env.step(action, model=_model) for env, action in zip(self.env_list, actions)]
        return results
        # try:
        #     obs, rews, dones, infos, cent_state = zip(*results)
        #     return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(infos), np.stack(cent_state)
        # except:
        #     obs, rews, dones, infos, cent_state, reject_rate, total_reward = zip(*results)
        #     return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(infos), np.stack(cent_state), reject_rate, total_reward

    def reset(self, _step_start, _model):
        results = [env.reset(step_start=_step_start, model=_model) for env in self.env_list]
        obs, infos, cent_state = zip(*results)
        return np.stack(obs), np.stack(infos), np.stack(cent_state)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass


# single env 
class DummyVecEnv(object):
    def __init__(self, all_args):
        """
        envs: list of gym environments to run in subprocesses
        for test
        """

        self.env_list = [Env(i) for i in range(all_args.n_eval_rollout_threads)]
        self.num_envs = all_args.n_rollout_threads

        self.num_agent = self.env_list[0].agent_num

        self.u_range = 1.0  # control range for continuous control
        self.movable = True

        # environment parameters
        self.discrete_action_space = True

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = False
        # in this env, force_discrete_action == False because world do not have discrete_action

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 145
        st_obs_dim = 3
        for agent_num in range(self.num_agent):
            total_action_space = []

            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(8) 
            else:
                u_action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(6000,), dtype=np.int8)  # [-1,1]
            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            obs_dim = 10 
            # share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))  # [-inf,inf]

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.num_agent)]
        self.st_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(st_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.num_agent)]

    def step(self, actions, _model):
        results = [env.step(action, model=_model) for env, action in zip(self.env_list, actions)]
        return results
        
    def reset(self, _step_start, _model):
        results = [env.reset(step_start=_step_start, mode="test", model=_model) for env in self.env_list]
        obs, infos, cent_state = zip(*results)
        return np.stack(obs), np.stack(infos), np.stack(cent_state)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass
