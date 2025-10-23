import os
from absl import app, flags
from flax import nnx
import gym
import copy
from tqdm import tqdm
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'antmaze-medium-play-v2', 'environment name')
flags.DEFINE_string('mode', 'train', 'Mode of operation')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_string('device', 'cuda:0', 'Device to use')
flags.DEFINE_string('project', 'pg', 'Project name')

flags.DEFINE_string('planner_net', 'transformer', 'Type of planner network')
flags.DEFINE_bool('rebase_policy', True, 'Rebase policy position') 

flags.DEFINE_bool('continous_reward_at_done', True, 'Continous reward at done')
flags.DEFINE_string('reward_tune', 'iql', 'Reward tune')
flags.DEFINE_float('discount', 1.0, 'Discount factor')
flags.DEFINE_integer('planner_d_model_divide', 64, 'divide planner model dimension')

flags.DEFINE_string('planner_solver', 'ddim', 'Planner solver')
flags.DEFINE_integer('planner_emb_dim', 128, 'Planner embedding dimension')
flags.DEFINE_integer('planner_d_model', 256, 'Planner model dimension')
flags.DEFINE_integer('planner_depth', 8, 'Planner depth')
flags.DEFINE_integer('planner_sampling_steps', 20, 'Planner sampling steps')
flags.DEFINE_bool('planner_predict_noise', True, 'Planner predict noise')
flags.DEFINE_float('planner_next_obs_loss_weight', 1, 'Planner next observation loss weight')
flags.DEFINE_float('planner_ema_rate', 0.9999, 'Planner EMA rate')
flags.DEFINE_integer('unet_dim', 64, 'UNet dimension')
flags.DEFINE_integer('use_advantage_weighting', 0, 'Use advantage weighting')

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
flags.DEFINE_integer('log_interval', 100000, 'Log interval')
flags.DEFINE_integer('eval_interval', 1000000, 'Save interval')

flags.DEFINE_integer('num_episodes', 100, 'Number of episodes')
flags.DEFINE_integer('planner_ckpt', 1000000, 'Planner checkpoint')
flags.DEFINE_integer('critic_ckpt', 200000, 'Critic checkpoint')
flags.DEFINE_integer('policy_ckpt', 1000000, 'Policy checkpoint')
flags.DEFINE_integer('invdyn_ckpt', 200000, 'Inverse dynamics checkpoint')
flags.DEFINE_bool('planner_use_ema', True, 'Use EMA for planner')
flags.DEFINE_float('policy_temperature', 0.5, 'Policy temperature')
flags.DEFINE_bool('policy_use_ema', True, 'Use EMA for policy')

flags.DEFINE_integer('max_path_length', 1000, 'Maximum path length')
flags.DEFINE_integer('planner_horizon', 40, 'Planner horizon')
flags.DEFINE_integer('stride', 25, 'Stride')
flags.DEFINE_float('planner_temperature', 1.0, 'Planner temperature')
flags.DEFINE_float('planner_target_return', 1.0, 'Planner target return')
flags.DEFINE_float('planner_w_cfg', 1.0, 'Planner weight for CFG')

# PG
flags.DEFINE_integer("prior_hidden_dim", 256, "Prior hidden dimensions (comma separated)")
flags.DEFINE_float('prior_squash_mean', 2.0, 'prior squash mean')
flags.DEFINE_float('prior_learning_rate', 3e-4, 'prior learning rate')
flags.DEFINE_float('alpha', 1.0, 'weight for const loss')
flags.DEFINE_bool('use_tanh_squash', True, 'use squash')
flags.DEFINE_string('divergence', 'kl', 'divergence type')
flags.DEFINE_bool('normalize_q', True, 'normalize q')
flags.DEFINE_bool('use_value', False, 'use value')
flags.DEFINE_bool('use_target', True, 'use target')
flags.DEFINE_bool('eval_deterministic', True, 'eval deterministic')
flags.DEFINE_integer('planner_sampling_steps_train', 5, 'Planner planner_sampling_steps_train steps')

flags.DEFINE_string("entity", "entity_name", "entity name")




def main(_):
    import pickle
    import hashlib
    import json
    import jax.numpy as jnp
    import jax
    from easydict import EasyDict
    import wandb
    import optax
    from dataset.d4rl_antmaze_dataset import D4RLAntmazeSeqDataset
    import pg
    from evaluation import evaluate_prior

    config = EasyDict(FLAGS.flag_values_dict())

    wandb.init(project=config.project, entity=config.entity)
    wandb.config.update(config)

    env = gym.make(config.env_name)
    planner_dataset = D4RLAntmazeSeqDataset(
        env.get_dataset(), horizon=config.planner_horizon, discount=config.discount, 
        continous_reward_at_done=config.continous_reward_at_done, reward_tune=config.reward_tune, 
        stride=config.stride, learn_policy=False, center_mapping=(config.guidance_type!="cfg")
    )

    config.observation_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]

    planner, policy, critic, value, agent_state = pg.init(config)
    update_fn = pg.update

    model_path = f"./checkpoint/DV/{config.env_name}/{config.seed}/model_{config.planner_ckpt}.pkl"
    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
  
    model_critic_path = f"./checkpoint/DV/{config.env_name}/{config.seed}/model_{config.critic_ckpt}.pkl"
    with open(model_critic_path, 'rb') as f:
        model_critic_info = pickle.load(f)
    
    target_planner_params = model_info['target_planner_params']
    target_policy_params = model_info['target_policy_params']
    critic_params = model_critic_info['critic']

    nnx.update(planner.model, target_planner_params)
    nnx.update(policy.model, target_policy_params)
    nnx.update(critic, critic_params)

    planner.model.eval()
    policy.model.eval()
    critic.eval()

    if config.mode == "train":
        config.use_planner_condition = False
        config.use_policy_condition = True
        config.train_condition = False 

        def train_step(carry, _):
            (agent_state, i) = carry
            agent_state, info = update_fn(planner, critic, None, agent_state, config, planner_dataset)
            return (agent_state, i + 1), info

        normalizer = planner_dataset.normalizer

        i = 0        
        carry = (agent_state, 0)

        while i < config.planner_diffusion_gradient_steps:
            carry, info = jax.lax.scan(train_step, carry, None, config.log_interval)

            (agent_state, i) = carry

            wandb.log({f"train/{k}": v.mean() for k, v in info._asdict().items() if v is not None}, step=i)

            if i % config.eval_interval == 0:
                prior, _ = nnx.merge(agent_state.prior_graphdef, agent_state.prior_state)
                prior.eval()

                nnx.update(prior, agent_state.target_prior_params)

                eval_info, rng = evaluate_prior(planner, policy, prior, agent_state.rng, config, env, normalizer, eval_deterministic=config.eval_deterministic)
                wandb.log({f"det_eval_target/{k}": v for k, v in eval_info.items()}, step=i)
                agent_state = agent_state._replace(rng=rng)

                carry = (agent_state, i)


if __name__ == '__main__':
    app.run(main)
