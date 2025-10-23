from optax import incremental_update
from collections import namedtuple
from flax import nnx
import jax
import jax.numpy as jnp
import copy
from easydict import EasyDict
import optax
from flax.training import train_state
import tensorflow_probability.substrates.jax as tfp
import numpy as np

from network import DAHorizonCritic, DAMlp, IdentityCondition, DiT1d, TanhStochasticGRU, V
from diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE


tfd = tfp.distributions
tfb = tfp.bijectors
prior_DV_state = namedtuple('prior_DV_state', ['prior_state', 'prior_graphdef', 'target_prior_params', 'critic_T_state', 'critic_T_graphdef', 'target_critic_T_params', 'value_T_state', 'value_T_graphdef', 'target_value_T_params', 'rng'])
prior_DV_info = namedtuple('prior_DV_info', ['total_loss', 'prior_loss', 'const_loss', 'abs_prior_mean', 'prior_mean', 'prior_std', 'critic_loss'])

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

class TrainState(train_state.TrainState):
    other_variables: nnx.State


def init(config: EasyDict):
    
    planner_dim = config.observation_dim if config.pipeline_type=="separate" else config.observation_dim + config.action_dim
    
    # --------------- Network Architecture -----------------
    nn_diffusion_planner = DiT1d(
        planner_dim, emb_dim=config.planner_emb_dim,
        d_model=config.planner_d_model, n_heads=config.planner_d_model//config.planner_d_model_divide, depth=config.planner_depth, timestep_emb_type="fourier",
        rngs=nnx.Rngs(config.seed))


    nn_condition_planner = None
    classifier = None
    

    critic = DAHorizonCritic(
        planner_dim, emb_dim=config.planner_emb_dim,
        d_model=config.planner_d_model, n_heads=config.planner_d_model//config.planner_d_model_divide, depth=2, norm_type="pre", rngs=nnx.Rngs(config.seed))

    # ----------------- Masking -------------------
    fix_mask = np.zeros((config.planner_horizon, planner_dim))
    fix_mask[0, :config.observation_dim] = 1.
    loss_weight = np.ones((config.planner_horizon, planner_dim))
    loss_weight[1] = config.planner_next_obs_loss_weight

    fix_mask = jnp.array(fix_mask)
    loss_weight = jnp.array(loss_weight) 

    # --------------- Diffusion Model with Classifier-Free Guidance --------------------
    planner = ContinuousDiffusionSDE(
        nn_diffusion_planner, nn_condition_planner,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=config.planner_ema_rate,
        predict_noise=config.planner_predict_noise, noise_schedule="linear", planner_diffusion_gradient_steps=config.planner_diffusion_gradient_steps)


    nn_diffusion_policy = DAMlp(config.observation_dim, config.action_dim, emb_dim=64, hidden_dim=config.policy_hidden_dim, timestep_emb_type="positional", rngs=nnx.Rngs(config.seed))
    nn_condition_policy = IdentityCondition(dropout=0.0)

    # --------------- Diffusion Model Actor --------------------
    policy = DiscreteDiffusionSDE(
        nn_diffusion_policy, nn_condition_policy, predict_noise=config.policy_predict_noise,
        x_max=+1. * jnp.ones((1, config.action_dim)),
        x_min=-1. * jnp.ones((1, config.action_dim)),
        diffusion_steps=config.policy_diffusion_steps, ema_rate=config.policy_ema_rate)


    # --------------- Prior --------------------
    prior = TanhStochasticGRU(config.observation_dim, planner_horizon=config.planner_horizon, hidden_dim=config.prior_hidden_dim, squash_mean=config.prior_squash_mean, divergence=config.divergence, rngs=nnx.Rngs(config.seed))
    prior_optimizer = nnx.Optimizer(prior, optax.adam(learning_rate=config.prior_learning_rate))
    
    # --------------- Value --------------------
    critic = DAHorizonCritic(
            planner_dim, emb_dim=config.planner_emb_dim,
            d_model=config.planner_d_model, n_heads=config.planner_d_model//config.planner_d_model_divide, depth=2, norm_type="pre", rngs=nnx.Rngs(config.seed))
    
    value = V(config.observation_dim, hidden_dim=256, rngs=nnx.Rngs(config.seed))


    # --------------- latent value --------------------
    critic_T = DAHorizonCritic(
            planner_dim, emb_dim=config.planner_emb_dim,
            d_model=config.planner_d_model, n_heads=config.planner_d_model//config.planner_d_model_divide, depth=2, norm_type="pre", rngs=nnx.Rngs(config.seed))
    critic_T_lr = optax.cosine_decay_schedule(config.critic_learning_rate, config.planner_diffusion_gradient_steps)
    critic_T_optimizer = nnx.Optimizer(critic_T, optax.adam(learning_rate=critic_T_lr))
    critic_T_graphdef, critic_T_state = nnx.split((critic_T, critic_T_optimizer))
    target_critic_T_params = critic_T_state.filter(nnx.Param)

    value_T = DAHorizonCritic(
            planner_dim, emb_dim=config.planner_emb_dim,
            d_model=config.planner_d_model, n_heads=config.planner_d_model//config.planner_d_model_divide, depth=2, norm_type="pre", rngs=nnx.Rngs(config.seed))
    value_T_lr = optax.cosine_decay_schedule(config.critic_learning_rate, config.planner_diffusion_gradient_steps)
    value_T_optimizer = nnx.Optimizer(value_T, optax.adam(learning_rate=value_T_lr))
    value_T_graphdef, value_T_state = nnx.split((value_T, value_T_optimizer))
    target_value_T_params = value_T_state.filter(nnx.Param)

    
    prior_graphdef, prior_state = nnx.split((prior, prior_optimizer))
    target_prior_params = copy.deepcopy(nnx.state(prior, nnx.Param))

    rng = jax.random.key(config.seed)

    return planner, policy, critic, value, prior_DV_state(
        prior_state=prior_state,
        prior_graphdef=prior_graphdef,
        target_prior_params=target_prior_params,
        critic_T_state=critic_T_state,
        critic_T_graphdef=critic_T_graphdef,
        target_critic_T_params=target_critic_T_params,
        value_T_state=value_T_state,
        value_T_graphdef=value_T_graphdef,
        target_value_T_params=target_value_T_params,
        rng=rng
    )

def prior_update(prior_DV_state, config, planner_batch):
    prior_state = prior_DV_state.prior_state
    prior_graphdef = prior_DV_state.prior_graphdef
    value_T_state = prior_DV_state.value_T_state
    value_T_graphdef = prior_DV_state.value_T_graphdef
    value_T, _ = nnx.merge(value_T_graphdef, value_T_state)
    critic_T_state = prior_DV_state.critic_T_state
    critic_T_graphdef = prior_DV_state.critic_T_graphdef
    critic_T, _ = nnx.merge(critic_T_graphdef, critic_T_state)

    if config.use_target:
        _, other_variables = value_T_state.split(nnx.Param, ...)
        _, other_variables_critic = critic_T_state.split(nnx.Param, ...)
        value_T = nnx.merge(value_T_graphdef, prior_DV_state.target_value_T_params, other_variables)[0]
        critic_T = nnx.merge(critic_T_graphdef, prior_DV_state.target_critic_T_params, other_variables_critic)[0]

    key1, rng = jax.random.split(prior_DV_state.rng, 2)
    obs_one = planner_batch.obs[:, 0, :]
    

    def loss_fn(prior):
        dist = prior(obs_one)
        prior_value = dist.sample(seed=key1)
        if config.use_tanh_squash:
            prior_value = jnp.tanh(prior_value) * config.prior_squash_mean

        info = {'abs_prior_mean': abs(prior_value).mean(), 'prior_mean': dist.mean().mean(), 'prior_std':dist.stddev().mean()}

        prior_s_concat = jnp.concatenate((jnp.expand_dims(obs_one, axis=1), prior_value), axis=1)

        if config.use_value:
            val = value_T(prior_s_concat)
            prior_loss = -val.mean()
        else:
            val = critic_T(prior_s_concat)
            prior_loss = -val.mean()

        if config.normalize_q:
            lmbda = jax.lax.stop_gradient(1 / jnp.abs(val).mean())
            prior_loss = prior_loss * lmbda

        std_normal_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(prior_value.shape), scale_diag=jnp.ones(prior_value.shape))
    
        if config.divergence == 'kl':
            kl = tfd.kl_divergence(dist, std_normal_dist)
            const_loss = kl.mean()
        else:
            NotImplementedError()

        total_loss = prior_loss + config.alpha * const_loss

        return total_loss, {'prior_loss':prior_loss, 'const_loss': const_loss, **info}
    
    
    prior, optimizer = nnx.merge(prior_graphdef, prior_state)
    (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(prior)
    optimizer.update(grads)
    _, prior_state = nnx.split((prior, optimizer))

    target_prior_params = incremental_update(nnx.state(prior, nnx.Param), prior_DV_state.target_prior_params, 0.005)

    prior_DV_state = prior_DV_state._replace(prior_state=prior_state, target_prior_params=target_prior_params, rng=rng)
    
    return prior_DV_state, {'total_loss': loss, **info}


def critic_update(planner, critic, value, prior_DV_state, config, planner_batch):
    value_T_state = prior_DV_state.value_T_state
    value_T_graphdef = prior_DV_state.value_T_graphdef
    critic_T_state = prior_DV_state.critic_T_state
    critic_T_graphdef = prior_DV_state.critic_T_graphdef
    prior_state = prior_DV_state.prior_state
    prior_graphdef = prior_DV_state.prior_graphdef
    
    key1, key2, rng = jax.random.split(prior_DV_state.rng, 3)
    obs_one = planner_batch.obs[:, 0, :]

    prior, _ = nnx.merge(prior_graphdef, prior_state)
    nnx.update(prior, prior_DV_state.target_prior_params)
    

    dist = prior(obs_one)
    prior_value = dist.sample(seed=key1)
    if config.use_tanh_squash:
        prior_value = jnp.tanh(prior_value) * config.prior_squash_mean

    traj = planner.sample_prior_train(obs_one, prior=prior_value, config=config, rng=key2)
    
    if config.use_value:
        value_td_ev = value(traj)[:, 1:]
        val_0 = value_td_ev.mean(axis=1)
        value_T, optimizer = nnx.merge(value_T_graphdef, value_T_state)
    else:
        val_0 = critic(traj)
        critic_T, optimizer = nnx.merge(critic_T_graphdef, critic_T_state)

    def critic_loss_fn(model):
        val_pred = model(jnp.concatenate((jnp.expand_dims(obs_one, axis=1), prior_value), axis=1))
        assert val_pred.shape == val_0.shape
        loss = ((val_pred - val_0)**2).mean()
        return loss
    
    def value_loss_fn(model):
        val_pred = model(jnp.concatenate((jnp.expand_dims(obs_one, axis=1), prior_value), axis=1))
        assert val_pred.shape == val_0.shape
        loss = ((val_pred - val_0)**2).mean()
        return loss
    
    if config.use_value:
        loss, grads = nnx.value_and_grad(value_loss_fn)(value_T)
        optimizer.update(grads)
        _, value_T_state = nnx.split((value_T, optimizer))
        target_value_T_params = incremental_update(value_T_state.filter(nnx.Param), prior_DV_state.target_value_T_params, 0.005)
        prior_DV_state = prior_DV_state._replace(value_T_state=value_T_state, target_value_T_params=target_value_T_params, rng=rng)
    else:
        loss, grads = nnx.value_and_grad(critic_loss_fn)(critic_T)
        optimizer.update(grads)
        _, critic_T_state = nnx.split((critic_T, optimizer))
        target_critic_T_params = incremental_update(critic_T_state.filter(nnx.Param), prior_DV_state.target_critic_T_params, 0.005)
        prior_DV_state = prior_DV_state._replace(critic_T_state=critic_T_state, target_critic_T_params=target_critic_T_params, rng=rng)

    return prior_DV_state, loss



def update(planner, critic, value, prior_DV_state, config, planner_dataset):
    key1, rng = jax.random.split(prior_DV_state.rng, 2)
    prior_DV_state = prior_DV_state._replace(rng=rng)
    planner_batch = planner_dataset.sample(key1, config.batch_size)

    prior_DV_state, critic_loss = critic_update(planner, critic, value, prior_DV_state, config, planner_batch)
    prior_DV_state, info = prior_update(prior_DV_state, config, planner_batch)

    return prior_DV_state, prior_DV_info(
        total_loss=info['total_loss'], 
        prior_loss=info['prior_loss'], 
        const_loss=info['const_loss'], 
        abs_prior_mean=info['abs_prior_mean'], 
        prior_mean=info['prior_mean'], 
        prior_std=info['prior_std'], 
        critic_loss=critic_loss
    )
