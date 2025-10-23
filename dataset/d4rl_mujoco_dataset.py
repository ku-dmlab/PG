import os
from typing import Tuple
import gym
import numpy as np
from tqdm import tqdm
import d4rl
import collections
import wrappers

import jax.numpy as jnp
import jax

from functools import partial
from typing import Dict
from util import at_least_ndim, GaussianNormalizer


Batch = collections.namedtuple(
    'Batch',
    ['obs', 'act', 'rew', 'val'])


class D4RLMuJoCoSeqDataset():
    """
     Args:
        dataset: Dict[str, np.ndarray],
            D4RL-MuJoCo dataset. Obtained by calling `env.get_dataset()`.
        terminal_penalty: float,
            Penalty reward for early-terminal states. Default is -100.
        horizon: int,
            Length of each sequence. Default is 1.
        max_path_length: int,
            Maximum length of the episodes. Default is 1000.
        discount: float,
            Discount factor. Default is 0.99.
    """
    def __init__(self,
                 dataset: Dict[str, np.ndarray],
                 terminal_penalty: float = -100,
                 horizon: int = 1,
                 max_path_length: int = 1000,
                 discount: float = 0.99,
                 center_mapping: bool = True,
                 stride: int = 1,
                 full_traj_bonus: float = 100,
                #  normalize: bool =True,
    ):
        observations, actions, rewards, timeouts, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["timeouts"].astype(np.float32),
            dataset["terminals"].astype(np.float32))
        self.stride = stride

        self.normalizer = GaussianNormalizer(dataset['observations'])
        normed_observations = self.normalizer.normalize(observations)

        self.horizon = horizon
        self.observation_dim, self.action_dim = observations.shape[-1], actions.shape[-1]

        n_paths = np.sum(np.logical_or(terminals, timeouts))

        self.seq_obs = np.zeros((n_paths+1, max_path_length, self.observation_dim), dtype=np.float32)
        self.seq_act = np.zeros((n_paths+1, max_path_length, self.action_dim), dtype=np.float32)
        self.seq_rew = np.zeros((n_paths+1, max_path_length, 1), dtype=np.float32)
        self.seq_val = np.zeros((n_paths+1, max_path_length, 1), dtype=np.float32)

        self.indices = []

        ptr = 0
        path_idx = 0
        for i in range(timeouts.shape[0]):
            if timeouts[i] or terminals[i] or i == timeouts.shape[0] - 1:
                path_length = i - ptr + 1
                assert path_length <= max_path_length, f"current path length {path_length}"

                if terminals[i]:
                    rewards[i] = terminal_penalty if terminal_penalty is not None else rewards[i]
                    # rewards = rewards.at[i].set(terminal_penalty) if terminal_penalty is not None else rewards.at[i].set(rewards[i])
                    
                if path_length == max_path_length:
                    rewards[i] = rewards[i] + full_traj_bonus if full_traj_bonus is not None else rewards[i]

                self.seq_obs[path_idx, :path_length] = normed_observations[ptr:i + 1]
                self.seq_act[path_idx, :path_length] = actions[ptr:i + 1]
                self.seq_rew[path_idx, :path_length] = rewards[ptr:i + 1][:, None]

                max_start = path_length - (horizon - 1) * stride - 1
                self.indices += [(path_idx, start, start + (horizon - 1) * stride + 1) for start in range(max_start + 1)]

                ptr = i + 1
                path_idx += 1

        self.seq_val[:, -1] = self.seq_rew[:, -1]
        for i in reversed(range(max_path_length-1)):
            self.seq_val[:, i] = self.seq_rew[:, i] + discount * self.seq_val[:, i+1]
        
        print(f"max discounted return: {self.seq_val.max()}")
        print(f"min discounted return: {self.seq_val.min()}")
        
        # val \in [-1, 1]
        self.seq_val = (self.seq_val - self.seq_val.min()) / (self.seq_val.max() - self.seq_val.min())
        if center_mapping:
            self.seq_val = self.seq_val * 2 - 1
        print(f"max normed discounted return: {self.seq_val.max()}")
        print(f"min normed discounted return: {self.seq_val.min()}")

        # Since slicing is difficult in jax, preprocess data slicing while in numpy
        preprocessed_data = {
            'obs': [],
            'act': [],
            'rew': [],
            'val': []
        }

        for path_idx, start, end in self.indices:
            horizon_state = self.seq_obs[path_idx, start:end:self.stride]
            preprocessed_data['obs'].append(horizon_state)
            preprocessed_data['act'].append(self.seq_act[path_idx, start:end:self.stride])
            preprocessed_data['rew'].append(self.seq_rew[path_idx, start:end:self.stride])
            preprocessed_data['val'].append(self.seq_val[path_idx, start])

        preprocessed_data['obs'] = jnp.array(preprocessed_data['obs'])
        preprocessed_data['act'] = jnp.array(preprocessed_data['act'])
        preprocessed_data['rew'] = jnp.array(preprocessed_data['rew'])
        preprocessed_data['val'] = jnp.array(preprocessed_data['val'])
    
        self.data = Batch(preprocessed_data['obs'], preprocessed_data['act'], preprocessed_data['rew'], preprocessed_data['val'])
        

    def sample(self, key, batch_size):
        return _sample(self.data, batch_size, key)
    

def get_pytree_batch_item(tree_batch, idx):
    return jax.tree_util.tree_map(lambda tb: tb[idx], tree_batch)

sample_fn = jax.vmap(get_pytree_batch_item,  in_axes=(None, 0))

@partial(jax.jit, static_argnums=(1))
def _sample(data, batch_size, key):
    size = data.obs.shape[0]
    idx = jax.random.randint(key, minval=0, maxval=size, shape=(batch_size,))
    return sample_fn(data, idx)


Batch_td = collections.namedtuple(
    'Batch_td',
    ['obs', 'act', 'rew', 'tml', 'next_obs'])

class D4RLMuJoCoTDDataset():
    def __init__(self, dataset: Dict[str, np.ndarray]):
        
        # iql normalization
        dataset['rewards'] = reward_normalize(dataset)

        observations, actions, next_observations, rewards, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["next_observations"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["terminals"].astype(np.float32))
        


        self.normalizer = GaussianNormalizer(observations)
        normed_observations = self.normalizer.normalize(observations)
        normed_next_observations = self.normalizer.normalize(next_observations)

        self.obs = jnp.array(normed_observations, dtype=jnp.float32)
        self.act = jnp.array(actions, dtype=jnp.float32)
        self.rew = jnp.array(rewards, dtype=jnp.float32)[:, None]
        self.tml = jnp.array(terminals, dtype=jnp.float32)[:, None]
        self.next_obs = jnp.array(normed_next_observations, dtype=jnp.float32)

        self.size = self.obs.shape[0]
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

        self.data = Batch_td(self.obs, self.act, self.rew, self.tml, self.next_obs)

    def sample(self, key, batch_size):
        return _sample(self.data, batch_size, key)
    

def get_pytree_batch_item(tree_batch, idx):
    return jax.tree_util.tree_map(lambda tb: tb[idx], tree_batch)

sample_fn = jax.vmap(get_pytree_batch_item,  in_axes=(None, 0))

@partial(jax.jit, static_argnums=(1))
def _sample(data, batch_size, key):
    size = data.obs.shape[0]
    idx = jax.random.randint(key, minval=0, maxval=size, shape=(batch_size,))
    return sample_fn(data, idx)


def reward_normalize(dataset):
    
    rewards = dataset['rewards']
    dones_float = np.zeros_like(dataset['rewards'])

    dones_float[-1] = 1
    dones_float[:-1] = (np.linalg.norm(dataset['observations'][1:] -
                            dataset['next_observations'][:-1], axis=1
                            ) > 1e-6) | (dataset['terminals'][:-1] == 1.0)
    
    dones_index = np.where(dones_float)
    reward_cumsum = rewards.cumsum()
    episode_return = reward_cumsum[dones_index][1:] - reward_cumsum[dones_index][:-1]
    first_epsisode_return = reward_cumsum[dones_index][0]
    
    max_return, min_return = max(first_epsisode_return, episode_return.max()), min(first_epsisode_return, episode_return.min())
    rewards = 1000 * rewards / (max_return - min_return)
    return rewards
