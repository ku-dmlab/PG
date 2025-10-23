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



class D4RLMaze2DSeqDataset():
    def __init__(
            self,
            dataset: Dict[str, np.ndarray],
            horizon: int = 1,
            max_path_length: int = 800,
            discount: float = 0.99,
            continous_reward_at_done: bool = False,
            reward_tune: str = "iql",
            center_mapping: bool = True,
            learn_policy: bool = False,
            stride: int = 1,
    ):

        self.max_path_length = max_path_length

        observations, actions, rewards, timeouts = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["timeouts"].astype(np.float32))
        self.learn_policy = learn_policy
        self.stride = stride

        self.normalizer = GaussianNormalizer(observations)
        normed_observations = self.normalizer.normalize(observations)

        self.horizon = horizon
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

        self.indices = []
        self.seq_obs, self.seq_act, self.seq_rew = [], [], []

        path_idx = 0
        
        self.paths = []
        next_end = [-1] * (timeouts.shape[0] + 1)
        next_start = [-1] * (timeouts.shape[0] + 1)
        for index in reversed(range(timeouts.shape[0])):
            if rewards[index] == 1.0:
                next_end[index] = index
                next_start[index] = next_start[index+1]
            else:
                next_end[index] = next_end[index+1]
                next_start[index] = index
                
        path_start = next_start[0]
        path_end = next_end[path_start]
        
        if self.learn_policy:
            for path_start in range(0, timeouts.shape[0], max_path_length):
                path_end = min(path_start + max_path_length - 1, timeouts.shape[0] - 1)
                path_length = path_end - path_start + 1
            
                _seq_obs = np.zeros((max_path_length + (horizon - 1) * stride, self.o_dim), dtype=np.float32)
                _seq_act = np.zeros((max_path_length + (horizon - 1) * stride, self.a_dim), dtype=np.float32)
                _seq_rew = np.zeros((max_path_length + (horizon - 1) * stride, 1), dtype=np.float32)
                
                _seq_obs[:path_length] = normed_observations[path_start:path_end+1]
                _seq_act[:path_length] = actions[path_start:path_end+1]
                _seq_rew[:path_length] = rewards[path_start:path_end+1][:, None]
                
                _seq_obs[path_length:] = normed_observations[path_end]  # repeat state
                _seq_act[path_length:] = 0 # zero action
                _seq_rew[path_length:] = 1 if continous_reward_at_done else 0
                
                self.seq_obs.append(_seq_obs)
                self.seq_act.append(_seq_act)
                self.seq_rew.append(_seq_rew)
                
                max_start = path_length - 1
                self.indices += [(path_idx, start, start + (horizon - 1) * stride + 1) for start in range(max_start + 1)]
                self.paths.append((path_start, path_end))
                path_idx += 1
                
        else:
            while path_end != -1:
                path_start = max(path_start, path_end - max_path_length + 1)
                path_length = path_end - path_start + 1
                assert path_length >= 2
                
                _seq_obs = np.zeros((max_path_length + (horizon - 1) * stride, self.o_dim), dtype=np.float32)
                _seq_act = np.zeros((max_path_length + (horizon - 1) * stride, self.a_dim), dtype=np.float32)
                _seq_rew = np.zeros((max_path_length + (horizon - 1) * stride, 1), dtype=np.float32)
                
                _seq_obs[:path_length] = normed_observations[path_start:path_end+1]
                _seq_act[:path_length] = actions[path_start:path_end+1]
                _seq_rew[:path_length] = rewards[path_start:path_end+1][:, None]
                
                _seq_obs[path_length:] = normed_observations[path_end]  # repeat state
                _seq_act[path_length:] = 0 # zero action
                _seq_rew[path_length:] = 1 if continous_reward_at_done else 0
                
                self.seq_obs.append(_seq_obs)
                self.seq_act.append(_seq_act)
                self.seq_rew.append(_seq_rew)
                
                max_start = path_length - 1
                self.indices += [(path_idx, start, start + (horizon - 1) * stride + 1) for start in range(max_start + 1)]
                self.paths.append((path_start, path_end))
                path_idx += 1
                
                path_start = next_start[path_end]
                path_end = next_end[path_start]
        
        self.seq_obs = np.array(self.seq_obs)
        self.seq_act = np.array(self.seq_act)
        self.seq_rew = np.array(self.seq_rew)
        
        if reward_tune == "iql":
            self.seq_rew += -1 
        elif reward_tune == "none":
            self.seq_rew = self.seq_rew
        else:
            raise ValueError(f"reward_tune: {reward_tune} is not supported.")
        
        self.seq_val = np.copy(self.seq_rew)
        print(self.seq_obs.shape)
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
            'val': [],
        }

        for path_idx, start, end in self.indices:
            if self.learn_policy:
                horizon_state = self.seq_obs[path_idx, start:end:self.stride].copy()
                horizon_state[:, :2] -= horizon_state[0, :2]
            else:
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