from typing import Dict

import gym
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp

# from jaxrl5.wrappers.wandb_video import WANDBVideo
def evaluate(planner, policy, critic, planner_model, policy_model, rng, config, env, normalizer):
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=config.num_episodes)
    progress = tqdm(range(config.num_episodes))
    
    episode_rewards = []
    for _ in progress:
        observation, cum_done, ep_reward, t = env.reset(), 0., 0., 0
        
        # for maze2d
        finished = np.zeros(1, dtype=bool)
        
        while not jnp.all(cum_done) and t < config.max_path_length + 1:
            observation = normalizer.normalize(observation)
            obs_repeat = jnp.repeat(jnp.expand_dims(observation, axis=0), config.planner_num_candidates, axis=0)

            key1, key2, rng= jax.random.split(rng, 3)
            
            # sample trajectory with planner
            traj = planner.sample(obs_repeat, model=planner_model, solver=config.planner_solver, planner_horizon=config.planner_horizon, 
                        n_samples=config.planner_num_candidates, sample_steps=config.planner_sampling_steps, temperature=config.planner_temperature, rng=key1)
            
            # MCSS
            value = critic.eval_forward(traj)
            idx = jnp.argmax(value)
            traj = traj[idx]
            traj = jnp.expand_dims(traj, axis=0)

            # sample action with policy
            policy_prior = jnp.zeros((1, config.action_dim))
            next_obs_plan = traj[:, 1, :]
            obs_policy = jnp.expand_dims(observation.clone(), axis=0)
            next_obs_policy = next_obs_plan.clone()
                        
            if config.rebase_policy:
                next_obs_policy = jnp.concatenate((next_obs_policy[:, :2] - obs_policy[:, :2], next_obs_policy[:, 2:]), axis=1)
                obs_policy = jnp.concatenate((jnp.zeros((obs_policy.shape[0], 2)), obs_policy[:, 2:]), axis=1)
            
            action = policy.sample(
                policy_prior,
                model=policy_model,
                rng=key2,
                solver=config.policy_solver,
                n_samples=1,
                sample_steps=config.policy_sampling_steps,
                condition_cfg=jnp.concatenate([obs_policy, next_obs_policy], axis=-1),
                temperature=config.policy_temperature)
            
            action = jnp.squeeze(action)
            # act = act.cpu().numpy()
            observation, reward, done, _ = env.step(action)

            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)
            
            if any(env in config.env_name for env in ["halfcheetah", "walker2d", "hopper"]):
                ep_reward += (reward * (1 - cum_done)) if t < config.max_path_length else reward
            elif any(env in config.env_name for env in ["antmaze", "kitchen"]):
                ep_reward += reward
            elif 'maze2d' in config.env_name:
                finished |= (reward == 1.0)
                ep_reward += finished
            else:
                NotImplementedError()
        
        if any(env in config.env_name for env in ["halfcheetah", "walker2d", "hopper"]):
            episode_rewards.append(ep_reward)
        elif 'antmaze' in config.env_name:
            episode_rewards.append(np.clip(ep_reward, 0., 1.))
        elif 'maze2d' in config.env_name:
            episode_rewards.append(ep_reward)
        elif 'kitchen' in config.env_name:
            episode_rewards.append(np.clip(ep_reward, 0., 4.))
        else:
            NotImplementedError()

        mean = np.mean(np.array(list(map(lambda x: env.get_normalized_score(x), episode_rewards))).reshape(-1) * 100)
        progress.set_description_str(str(mean))
    
    episode_rewards = list(map(lambda x: env.get_normalized_score(x), episode_rewards))
    episode_rewards = np.array(episode_rewards).reshape(-1) * 100
    mean = np.mean(episode_rewards)
    err = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
    print(mean, err)

    return {"return": mean, "return_std": err}, rng



def evaluate_prior(planner, policy, prior, rng, config, env, normalizer, eval_deterministic=True):
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=config.num_episodes)
    progress = tqdm(range(config.num_episodes))
    
    episode_rewards = []
    for _ in progress:
        observation, cum_done, ep_reward, t = env.reset(), 0., 0., 0
        
        # for maze2d
        finished = np.zeros(1, dtype=bool)
        
        while not jnp.all(cum_done) and t < config.max_path_length + 1:
            observation = normalizer.normalize(observation)
            obs_repeat = jnp.repeat(jnp.expand_dims(observation, axis=0), 1, axis=0)

            key1, key2, key3, rng = jax.random.split(rng, 4)

            dist = prior.eval_forward(obs_repeat)
            prior_value = dist.sample(seed=key3)

            if eval_deterministic:
                prior_value = dist.mean()
            if config.use_tanh_squash:
                prior_value = jnp.tanh(prior_value) * config.prior_squash_mean
      
            traj = planner.sample_prior(obs_repeat, prior=prior_value, solver=config.planner_solver, planner_horizon=config.planner_horizon, 
                        sample_steps=config.planner_sampling_steps, temperature=config.planner_temperature, rng=key1)

            policy_prior = jnp.zeros((1, config.action_dim))
            next_obs_plan = traj[:, 1, :]
            obs_policy = jnp.expand_dims(observation.clone(), axis=0)
            next_obs_policy = next_obs_plan.clone()
                        
            if config.rebase_policy:
                next_obs_policy = jnp.concatenate((next_obs_policy[:, :2] - obs_policy[:, :2], next_obs_policy[:, 2:]), axis=1)
                obs_policy = jnp.concatenate((jnp.zeros((obs_policy.shape[0], 2)), obs_policy[:, 2:]), axis=1)
            
            action = policy.sample(
                policy_prior,
                model=policy.model,
                rng=key2,
                solver=config.policy_solver,
                n_samples=1,
                sample_steps=config.policy_sampling_steps,
                condition_cfg=jnp.concatenate([obs_policy, next_obs_policy], axis=-1),
                temperature=config.policy_temperature)
            
            action = jnp.squeeze(action)
            observation, reward, done, _ = env.step(action)

            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)
            
            if any(env in config.env_name for env in ["halfcheetah", "walker2d", "hopper"]):
                ep_reward += (reward * (1 - cum_done)) if t < config.max_path_length else reward
            elif any(env in config.env_name for env in ["antmaze", "kitchen"]):
                ep_reward += reward
            elif 'maze2d' in config.env_name:
                finished |= (reward == 1.0)
                ep_reward += finished
            else:
                NotImplementedError()
        
        if any(env in config.env_name for env in ["halfcheetah", "walker2d", "hopper"]):
            episode_rewards.append(ep_reward)
        elif 'antmaze' in config.env_name:
            episode_rewards.append(np.clip(ep_reward, 0., 1.))
        elif 'maze2d' in config.env_name:
            episode_rewards.append(ep_reward)
        elif 'kitchen' in config.env_name:
            episode_rewards.append(np.clip(ep_reward, 0., 4.))
        else:
            NotImplementedError()
            
        mean = np.mean(np.array(list(map(lambda x: env.get_normalized_score(x), episode_rewards))).reshape(-1) * 100)
        progress.set_description_str(str(mean))
    
    episode_rewards = list(map(lambda x: env.get_normalized_score(x), episode_rewards))
    episode_rewards = np.array(episode_rewards).reshape(-1) * 100
    mean = np.mean(episode_rewards)
    err = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
    print(mean, err)

    return {"return": mean, "return_std": err}, rng
