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



class D4RLKitchenSeqDataset():
    def __init__(
            self,
            dataset: Dict[str, np.ndarray],
            horizon: int = 1,
            max_path_length: int = 280,
            discount: float = 0.99,
            center_mapping: bool = True,
            stride: int = 1,
    ):
        super().__init__()

        observations, actions, rewards, timeouts, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["timeouts"].astype(np.float32),
            dataset["terminals"].astype(np.float32))
        self.stride = stride

        self.normalizer = GaussianNormalizer(observations)
        normed_observations = self.normalizer.normalize(observations)

        self.horizon = horizon
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

        self.indices = []
        self.seq_obs, self.seq_next_obs, self.seq_act, self.seq_rew = [], [], [], []

        self.path_lengths, ptr = [], 0
        path_idx = 0
        for i in range(timeouts.shape[0]):
            if timeouts[i] or terminals[i] or i == timeouts.shape[0] - 1:
                self.path_lengths.append(i - ptr + 1)

                _seq_obs = np.zeros((max_path_length, self.o_dim), dtype=np.float32)
                _seq_next_obs = np.zeros((max_path_length, self.o_dim), dtype=np.float32)
                _seq_act = np.zeros((max_path_length, self.a_dim), dtype=np.float32)
                _seq_rew = np.zeros((max_path_length, 1), dtype=np.float32)

                _seq_obs[:i - ptr + 1] = normed_observations[ptr:i + 1]
                _seq_act[:i - ptr + 1] = actions[ptr:i + 1]
                _seq_rew[:i - ptr + 1] = rewards[ptr:i + 1][:, None]

                # repeat padding
                _seq_obs[i - ptr + 1:] = normed_observations[i]  # repeat last state
                _seq_act[i - ptr + 1:] = 0  # repeat zero action
                _seq_rew[i - ptr + 1:] = rewards[i]  # repeat last reward

                self.seq_obs.append(_seq_obs)
                self.seq_next_obs.append(_seq_next_obs)
                self.seq_act.append(_seq_act)
                self.seq_rew.append(_seq_rew)

                max_start = min(self.path_lengths[-1] - 1, max_path_length - (horizon - 1) * stride - 1)
                self.indices += [(path_idx, start, start + (horizon - 1) * stride + 1) for start in range(max_start + 1)]

                ptr = i + 1
                path_idx += 1

        self.seq_obs = np.array(self.seq_obs)
        self.seq_act = np.array(self.seq_act)
        self.seq_rew = np.array(self.seq_rew)

        self.seq_val = np.copy(self.seq_rew)
        for i in reversed(range(max_path_length - 1)):
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