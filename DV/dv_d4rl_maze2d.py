import os
from absl import app, flags
from flax import nnx
import gym
import copy
from tqdm import tqdm
import numpy as np
import sys
sys.path.append('.')

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'maze2d-umaze-v1', 'environment name') 
flags.DEFINE_string('mode', 'train', 'Mode of operation')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_string('device', 'cuda:0', 'Device to use')
flags.DEFINE_string('project', 'dv', 'Project name')

flags.DEFINE_string('guidance_type', 'MCSS', 'Type of guidance')
flags.DEFINE_string('planner_net', 'transformer', 'Type of planner network')
flags.DEFINE_string('pipeline_type', 'separate', 'Type of pipeline')
flags.DEFINE_bool('rebase_policy', True, 'Rebase policy position') 

flags.DEFINE_bool('continous_reward_at_done', True, 'Continous reward at done')
flags.DEFINE_string('reward_tune', 'iql', 'Reward tune')
flags.DEFINE_float('discount', 1.0, 'Discount factor')
flags.DEFINE_integer('planner_d_model_divide', 64, 'divide planner model dimension')

flags.DEFINE_string('planner_solver', 'ddim', 'Planner solver')
flags.DEFINE_integer('planner_emb_dim', 128, 'Planner embedding dimension')
flags.DEFINE_integer('planner_d_model', 256, 'Planner model dimension')
flags.DEFINE_integer('planner_depth', 2, 'Planner depth') 
flags.DEFINE_integer('planner_sampling_steps', 20, 'Planner sampling steps')
flags.DEFINE_bool('planner_predict_noise', True, 'Planner predict noise')
flags.DEFINE_float('planner_next_obs_loss_weight', 1, 'Planner next observation loss weight')
flags.DEFINE_float('planner_ema_rate', 0.9999, 'Planner EMA rate')
flags.DEFINE_integer('unet_dim', 32, 'UNet dimension') 
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
flags.DEFINE_integer('log_interval', 10000, 'Log interval')
flags.DEFINE_integer('save_interval', 200000, 'Save interval')

flags.DEFINE_integer('num_episodes', 100, 'Number of episodes') 
flags.DEFINE_integer('planner_num_candidates', 50, 'Number of planner candidates')
flags.DEFINE_integer('planner_ckpt', 1000000, 'Planner checkpoint')
flags.DEFINE_integer('critic_ckpt', 200000, 'Critic checkpoint')
flags.DEFINE_integer('policy_ckpt', 1000000, 'Policy checkpoint')
flags.DEFINE_integer('invdyn_ckpt', 200000, 'Inverse dynamics checkpoint')
flags.DEFINE_bool('planner_use_ema', True, 'Use EMA for planner')
flags.DEFINE_float('policy_temperature', 0.5, 'Policy temperature')
flags.DEFINE_bool('policy_use_ema', True, 'Use EMA for policy')

flags.DEFINE_integer('max_path_length', 800, 'Maximum path length')
flags.DEFINE_integer('planner_horizon', 32, 'Planner horizon') 
flags.DEFINE_integer('stride', 15, 'Stride') 
flags.DEFINE_float('planner_temperature', 1.0, 'Planner temperature')
flags.DEFINE_float('planner_target_return', 1.0, 'Planner target return')
flags.DEFINE_float('planner_w_cfg', 1.0, 'Planner weight for CFG')

# wandb
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
    from dataset.d4rl_maze2d_dataset import D4RLMaze2DSeqDataset
    import DV
    from evaluation import evaluate

    config = EasyDict(FLAGS.flag_values_dict())
    wandb.init(project=config.project, entity=config.entity)
    wandb.config.update(config)

    if config.env_name == 'maze2d-umaze-v1':
        config.max_path_length = 300
    elif config.env_name == 'maze2d-medium-v1':
        config.max_path_length = 600
    elif config.env_name == 'maze2d-large-v1':
        config.max_path_length = 800
    else:
        raise ValueError('Unknown environment name')


    env = gym.make(config.env_name)
    planner_dataset = D4RLMaze2DSeqDataset(
        env.get_dataset(), horizon=config.planner_horizon, discount=config.discount, 
        continous_reward_at_done=config.continous_reward_at_done, reward_tune=config.reward_tune, 
        stride=config.stride, learn_policy=False, center_mapping=(config.guidance_type!="cfg")
    )
    policy_dataset = D4RLMaze2DSeqDataset(
        env.get_dataset(), horizon=config.planner_horizon, discount=config.discount, 
        continous_reward_at_done=config.continous_reward_at_done, reward_tune=config.reward_tune, 
        stride=config.stride, learn_policy=True, center_mapping=(config.guidance_type!="cfg")
    )

    config.observation_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]

    planner, policy, agent_state = DV.init(config)
    update_fn = DV.update

    # ---------------------- Training ----------------------
    if config.mode == "train":
        config.use_planner_condition = False
        config.use_policy_condition = True
        config.train_condition = True 

        def train_step(carry, _):
            (agent_state, i) = carry
            agent_state, info = update_fn(planner, policy, agent_state, config, planner_dataset, policy_dataset)
            return (agent_state, i + 1), info

        i = 0
        carry = (agent_state, 0)

        while i < config.planner_diffusion_gradient_steps:
            carry, info = jax.lax.scan(train_step, carry, None, config.log_interval)
            (agent_state, i) = carry

            wandb.log({f"train/{k}": v.mean() for k, v in info._asdict().items() if v is not None}, step=i)
            
            # save model
            if i % config.save_interval == 0:
                checkpoint_dir = f'./checkpoint/DV/{config.env_name}/{config.seed}/'
                os.makedirs(checkpoint_dir, exist_ok=True)
                model_path = os.path.join(checkpoint_dir, f'model_{i}.pkl')
                critic, _ = nnx.merge(agent_state.critic_graphdef, agent_state.critic_state)
                planner_model, _ = nnx.merge(agent_state.planner_model_graphdef, agent_state.planner_model_state) 
                policy_model, _ = nnx.merge(agent_state.policy_model_graphdef, agent_state.policy_model_state) 

                nnx.update(planner_model, agent_state.target_planner_params)
                nnx.update(policy_model, agent_state.target_policy_params)


                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'critic': nnx.state(critic, nnx.Param),
                        'planner_model': nnx.state(planner_model, nnx.Param),
                        'policy_model': nnx.state(policy_model, nnx.Param),
                        'target_critic_params': agent_state.target_critic_params,
                        'target_planner_params': agent_state.target_planner_params,
                        'target_policy_params': agent_state.target_policy_params,
                    }, f)


        # last evaluation
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
        normalizer = planner_dataset.normalizer

        eval_info, rng = evaluate(planner, policy, critic, planner.model, policy.model, agent_state.rng, config, env, normalizer)
        wandb.log({f"eval_final/{k}": v for k, v in eval_info.items()}, step=0)


if __name__ == '__main__':
    app.run(main)

