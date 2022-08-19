import torch
import numpy as np
from utils.util import get_shape_from_obs_space, get_shape_from_act_space
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

class RLDataset(IterableDataset):
    """
    Args:
        buffer: replay buffer
    """

    def __init__(self, buffer, advantages, num_mini_batch):
        self.buffer = buffer
        self.advantages = advantages
        self.num_mini_batch = num_mini_batch

    def __iter__(self):
        share_obs_batch, obs_batch, actions_batch,\
            value_preds_batch, return_batch, st_value_preds_batch, st_return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
            adv_targ, available_actions_batch = self.buffer.feed_forward_generator(self.advantages, self.num_mini_batch)
        for i in range(len(obs_batch)):
            yield share_obs_batch[i], obs_batch[i], actions_batch[i],\
                  value_preds_batch[i], return_batch[i], st_value_preds_batch[i], st_return_batch[i], masks_batch[i], active_masks_batch[i], old_action_log_probs_batch[i],\
                  adv_targ[i], available_actions_batch[i]

def train_dataloader(buffer, advantages, num_mini_batch, batch_size):
    """Initialize the Replay Buffer dataset used for retrieving experiences."""
    dataset = RLDataset(buffer, advantages, num_mini_batch)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True
    )
    return dataloader


class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, st_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)
        st_obs_shape = get_shape_from_obs_space(st_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]
        
        if type(st_obs_shape[-1]) == list:
            st_obs_shape = st_obs_shape[:1]

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape),
                                  dtype=np.float32)
        # (109,1,6000,60000) > (109,1,6000,145)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)
        # (109,1,6000,10)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)

        self.st_value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.st_returns = np.zeros_like(self.st_value_preds)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n),
                                             dtype=np.int8)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.int8)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.int8)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(self, share_obs, obs, actions, action_log_probs,
               value_preds, st_value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step + 1] = share_obs #.copy()
        self.obs[self.step + 1] = obs #.copy()
        self.actions[self.step] = actions #.copy()
        self.action_log_probs[self.step] = action_log_probs #.copy()
        self.value_preds[self.step] = value_preds #.copy()
        self.st_value_preds[self.step] = st_value_preds #.copy()
        rewards = np.expand_dims(rewards, axis=2)
        self.rewards[self.step] = rewards #.copy()
        self.masks[self.step + 1] = masks #.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks #.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks #.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions #.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1] #.copy()
        self.obs[0] = self.obs[-1] #.copy()
        self.masks[0] = self.masks[-1] #.copy()
        self.bad_masks[0] = self.bad_masks[-1] #.copy()
        self.active_masks[0] = self.active_masks[-1] #.copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1] #.copy()

    def compute_returns(self, next_value, st_next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                self.st_value_preds[-1] = st_next_value
                st_gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                        st_delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.st_value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.st_value_preds[step])
                        st_gae = st_delta + self.gamma * self.gae_lambda * self.masks[step + 1] * st_gae
                        self.st_returns[step] = st_gae + value_normalizer.denormalize(self.st_value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def get_batch_size(self, num_mini_batch):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        mini_batch_size = batch_size // num_mini_batch
        return mini_batch_size
    
    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        sample training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents,
                          n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        # rand = torch.randperm(batch_size).numpy() 
        # sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        indices = torch.randint(batch_size, size=(mini_batch_size,), requires_grad=False)

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        st_value_preds = self.st_value_preds[:-1].reshape(-1, 1)
        st_returns = self.st_returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        share_obs_batch = share_obs[indices]
        obs_batch = obs[indices]
        actions_batch = actions[indices]
        if self.available_actions is not None:
            available_actions_batch = available_actions[indices]
        else:
            available_actions_batch = None
        value_preds_batch = value_preds[indices]
        return_batch = returns[indices]
        st_value_preds_batch = st_value_preds[indices]
        st_return_batch = st_returns[indices]
        masks_batch = masks[indices]
        active_masks_batch = active_masks[indices]
        old_action_log_probs_batch = action_log_probs[indices]
        if advantages is None:
            adv_targ = None
        else:
            adv_targ = advantages[indices]

        return share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, st_value_preds_batch, st_return_batch,masks_batch,\
                active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch