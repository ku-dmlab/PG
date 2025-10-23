import os
from absl import app, flags
from flax import nnx
import gym
import copy
from tqdm import tqdm
import numpy as np
from optax import incremental_update
import sys
sys.path.append('.')

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'walker2d-medium-v2', 'environment name')
flags.DEFINE_string('mode', 'train', 'Mode of operation')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_string('device', 'cuda:0', 'Device to use')
flags.DEFINE_string('project', 'dv', 'Project name')


flags.DEFINE_string('guidance_type', 'MCSS', 'Type of guidance')
flags.DEFINE_string('planner_net', 'transformer', 'Type of planner network')
flags.DEFINE_string('pipeline_type', 'separate', 'Type of pipeline')
flags.DEFINE_bool('rebase_policy', False, 'Rebase policy position')

flags.DEFINE_float('terminal_penalty', -100, 'Terminal penalty')
flags.DEFINE_float('full_traj_bonus', 0, 'Full trajectory bonus')
flags.DEFINE_float('discount', 0.997, 'Discount factor')

flags.DEFINE_integer('planner_d_model_divide', 32, 'divide planner model dimension')

flags.DEFINE_string('planner_solver', 'ddim', 'Planner solver')
flags.DEFINE_integer('planner_emb_dim', 128, 'Planner embedding dimension')
flags.DEFINE_integer('planner_d_model', 256, 'Planner model dimension')
flags.DEFINE_integer('planner_depth', 2, 'Planner depth')
flags.DEFINE_integer('planner_sampling_steps', 20, 'Planner sampling steps')

flags.DEFINE_bool('planner_predict_noise', True, 'Planner predict noise')
flags.DEFINE_float('planner_next_obs_loss_weight', 1, 'Planner next observation loss weight')
flags.DEFINE_float('planner_ema_rate', 0.9999, 'Planner EMA rate')
flags.DEFINE_integer('unet_dim', 32, 'UNet dimension')
flags.DEFINE_integer('use_advantage_weighting', 1, 'Use advantage weighting')
flags.DEFINE_integer('weight_factor', 2, 'Weight factor')

flags.DEFINE_string('policy_solver', 'ddpm', 'Policy solver')
flags.DEFINE_integer('policy_hidden_dim', 256, 'Policy hidden dimension')
flags.DEFINE_integer('policy_diffusion_steps', 10, 'Policy diffusion steps')
flags.DEFINE_integer('policy_sampling_steps', 10, 'Policy sampling steps')
flags.DEFINE_bool('policy_predict_noise', True, 'Policy predict noise')
flags.DEFINE_float('policy_ema_rate', 0.995, 'Policy EMA rate')
flags.DEFINE_float('policy_learning_rate', 0.0003, 'Policy learning rate')
flags.DEFINE_float('critic_learning_rate', 0.0003, 'Critic learning rate')

flags.DEFINE_integer('use_diffusion_policy', 1, 'Use diffusion policy')
flags.DEFINE_integer('invdyn_gradient_steps', 200000, 'Inverse dynamics gradient steps')
flags.DEFINE_integer('policy_diffusion_gradient_steps', 1000000, 'Policy diffusion gradient steps')
flags.DEFINE_integer('planner_diffusion_gradient_steps', 1000000, 'Planner diffusion gradient steps')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('log_interval', 10000, 'Log interval')
flags.DEFINE_integer('save_interval', 1000000, 'Save interval')

flags.DEFINE_integer('num_episodes', 10, 'Number of episodes')
flags.DEFINE_integer('planner_num_candidates', 50, 'Number of planner candidates')
flags.DEFINE_integer('planner_ckpt', 1000000, 'Planner checkpoint')
flags.DEFINE_integer('critic_ckpt', 200000, 'Critic checkpoint')
flags.DEFINE_integer('policy_ckpt', 1000000, 'Policy checkpoint')
flags.DEFINE_integer('invdyn_ckpt', 200000, 'Inverse dynamics checkpoint')
flags.DEFINE_bool('planner_use_ema', True, 'Use EMA for planner')
flags.DEFINE_float('policy_temperature', 0.5, 'Policy temperature')
flags.DEFINE_bool('policy_use_ema', True, 'Use EMA for policy')

flags.DEFINE_integer('max_path_length', 1000, 'Maximum path length')
flags.DEFINE_integer('planner_horizon', 4, 'Planner horizon')
flags.DEFINE_integer('stride', 1, 'Stride')
flags.DEFINE_float('planner_temperature', 1.0, 'Planner temperature')
flags.DEFINE_float('planner_target_return', 1.0, 'Planner target return')
flags.DEFINE_float('planner_w_cfg', 1.0, 'Planner weight for CFG')

# wandb
flags.DEFINE_string("entity", "entity_name", "entity name")


from collections import namedtuple
val_state = namedtuple('val_state', ['value_state', 'value_graphdef', 'target_value_params', 'rng'])


def main(_):
    import pickle
    import hashlib
    import json
    import jax.numpy as jnp
    import jax
    from easydict import EasyDict
    import wandb
    import optax
    from dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
    from network import V
    import d4rl

    config = EasyDict(FLAGS.flag_values_dict())
    wandb.init(project=config.project, entity=config.entity)
    wandb.config.update(config)

    env = gym.make(config.env_name)
    dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env))

    config.observation_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]


    if config.mode == "train":
        
        value_learning_rate = 3e-4

        value = V(config.observation_dim, hidden_dim=256, rngs=nnx.Rngs(config.seed))
        value_optimizer = nnx.Optimizer(value, optax.adam(learning_rate=value_learning_rate))
        value_graphdef, value_state = nnx.split((value, value_optimizer))
        target_value_params = value_state.filter(nnx.Param)

        agent_state = val_state(value_state=value_state, value_graphdef=value_graphdef, target_value_params=target_value_params, rng=jax.random.key(config.seed))

        def update_fn(agent_state, config, dataset):
            key1, rng = jax.random.split(agent_state.rng, 2)
            batch = dataset.sample(key1, config.batch_size)
            value_state = agent_state.value_state
            value_graphdef = agent_state.value_graphdef
            target_value_params = agent_state.target_value_params

            value, optimizer = nnx.merge(value_graphdef, value_state)
            _, other_variables = value_state.split(nnx.Param, ...)
            target_value = nnx.merge(value_graphdef, target_value_params, other_variables)[0]

            current_v = value(batch.obs)
            next_v =  target_value(batch.next_obs)

            def loss_fn(value):
                current_v = value(batch.obs)
                target_v = (batch.rew + (1 - batch.tml) * config.discount * next_v)
                loss = ((current_v - target_v)**2).mean()
                return loss
            
            loss, grads = nnx.value_and_grad(loss_fn)(value)
            optimizer.update(grads)
            _, value_state = nnx.split((value, optimizer))

            target_value_params = incremental_update(value_state.filter(nnx.Param), agent_state.target_value_params, 5e-3)

            agent_state = agent_state._replace(value_state=value_state, rng=rng, target_value_params=target_value_params)

            return agent_state, {'loss': loss, 'v': current_v, 'next_v': next_v}


        def train_step(carry, _):
            (agent_state, i) = carry
            agent_state, info = update_fn(agent_state, config, dataset)
            return (agent_state, i + 1), info 
        
        i = 0        
        carry = (agent_state, 0)
        model_path =  f'./checkpoint/DV/{config.env_name}/{config.seed}/'
        os.makedirs(model_path, exist_ok=True)

        while i < config.planner_diffusion_gradient_steps:
            carry, info = jax.lax.scan(train_step, carry, None, config.log_interval)
            (agent_state, i) = carry

            wandb.log({f"train/{k}": v.mean() for k, v in info.items() if v is not None}, step=i)

            if i % config.save_interval == 0:
                model_path = os.path.join(model_path, f'value_{i}.pkl')
                with open(model_path, 'wb') as f:
                        pickle.dump({
                            'value': agent_state.value_state.filter(nnx.Param),
                            'target_value_params': agent_state.target_value_params,
                        }, f)




if __name__ == '__main__':
    app.run(main)
