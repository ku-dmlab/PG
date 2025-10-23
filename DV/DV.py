from optax import incremental_update
from collections import namedtuple
from flax import nnx
import jax
import jax.numpy as jnp
import copy
from easydict import EasyDict
from functools import partial
import pickle
import optax
from flax.training import train_state
import tensorflow_probability.substrates.jax as tfp
import numpy as np

from network import DAHorizonCritic, DAMlp, IdentityCondition, DiT1d
from diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE

DV_state = namedtuple('DV_state', ['planner_model_state', 'planner_model_graphdef', 'critic_state', 'critic_graphdef', 'policy_model_state', 'policy_model_graphdef', 'target_critic_params', 'target_planner_params', 'target_policy_params', 'rng'])
DV_info = namedtuple('DV_info', ['planner_loss', 'critic_loss', 'policy_loss'])

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

class TrainState(train_state.TrainState):
    other_variables: nnx.State


def init(config: EasyDict):
    
    planner_dim = config.observation_dim if config.pipeline_type=="separate" else config.observation_dim + config.action_dim
    
    # --------------- Network Architecture -----------------
    if config.planner_net == "transformer":
        nn_diffusion_planner = DiT1d(
            planner_dim, emb_dim=config.planner_emb_dim,
            d_model=config.planner_d_model, n_heads=config.planner_d_model//config.planner_d_model_divide, depth=config.planner_depth, timestep_emb_type="fourier",
            rngs=nnx.Rngs(config.seed))
    else:
        NotImplementedError()

    nn_condition_planner = None
    classifier = None
    

    if config.guidance_type == "MCSS":
        # --------------- Horizon Critic -----------------
        critic = DAHorizonCritic(
            planner_dim, emb_dim=config.planner_emb_dim,
            d_model=config.planner_d_model, n_heads=config.planner_d_model//config.planner_d_model_divide, depth=2, norm_type="pre", rngs=nnx.Rngs(config.seed))
        critic_lr = optax.cosine_decay_schedule(config.critic_learning_rate, config.planner_diffusion_gradient_steps)
        critic_optimizer = nnx.Optimizer(critic, optax.adam(learning_rate=critic_lr))
    
    # ----------------- Masking -------------------
    fix_mask = np.zeros((config.planner_horizon, planner_dim))
    fix_mask[0, :config.observation_dim] = 1.
    loss_weight = np.ones((config.planner_horizon, planner_dim))
    loss_weight[1] = config.planner_next_obs_loss_weight

    fix_mask = jnp.array(fix_mask)
    loss_weight = jnp.array(loss_weight) 

    # --------------- Diffusion Model with Classifier-Free Guidance --------------------
    diff_lr = 2e-4
    weight_decay = 1e-5

    planner = ContinuousDiffusionSDE(
        nn_diffusion_planner, nn_condition_planner,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=config.planner_ema_rate,
        predict_noise=config.planner_predict_noise, noise_schedule="linear", planner_diffusion_gradient_steps=config.planner_diffusion_gradient_steps)
    
    planner_lr = optax.cosine_decay_schedule(diff_lr, config.planner_diffusion_gradient_steps)
    planner_optimizer = nnx.Optimizer(planner.model, optax.adamw(learning_rate=planner_lr, weight_decay=weight_decay))


    if config.pipeline_type=="separate":
        if config.use_diffusion_policy:
            nn_diffusion_policy = DAMlp(config.observation_dim, config.action_dim, emb_dim=64, hidden_dim=config.policy_hidden_dim, timestep_emb_type="positional", rngs=nnx.Rngs(config.seed))
            nn_condition_policy = IdentityCondition(dropout=0.0)
    
            # --------------- Diffusion Model Actor --------------------
            policy = DiscreteDiffusionSDE(
                nn_diffusion_policy, nn_condition_policy, predict_noise=config.policy_predict_noise,
                x_max=+1. * jnp.ones((1, config.action_dim)),
                x_min=-1. * jnp.ones((1, config.action_dim)),
                diffusion_steps=config.policy_diffusion_steps, ema_rate=config.policy_ema_rate)
            policy_lr = optax.cosine_decay_schedule(config.policy_learning_rate, config.policy_diffusion_gradient_steps)
            policy_optimizer = nnx.Optimizer(policy.model, optax.adamw(learning_rate=policy_lr))
        else:
            NotImplementedError()
    else:
        NotImplementedError()
    
    planner_model_graphdef, planner_model_state = nnx.split((planner.model, planner_optimizer))
    critic_graphdef, critic_state = nnx.split((critic, critic_optimizer))
    policy_model_graphdef, policy_model_state = nnx.split((policy.model, policy_optimizer))

    # ema model
    target_planner_params = copy.deepcopy(nnx.state(planner.model, nnx.Param))
    target_critic_params = copy.deepcopy(nnx.state(critic, nnx.Param))
    target_policy_params = copy.deepcopy(nnx.state(policy.model, nnx.Param))
    rng = jax.random.key(config.seed)

    return planner, policy, DV_state(
        planner_model_state=planner_model_state,
        planner_model_graphdef=planner_model_graphdef,
        critic_state=critic_state,
        critic_graphdef=critic_graphdef,
        policy_model_state=policy_model_state,
        policy_model_graphdef=policy_model_graphdef,
        target_planner_params=target_planner_params,
        target_critic_params=target_critic_params,
        target_policy_params=target_policy_params,
        rng=rng
    )


def critic_update(DV_state, config, planner_batch):
    critic_state = DV_state.critic_state
    critic_graphdef = DV_state.critic_graphdef
    rng = DV_state.rng

    def loss_fn(critic):
        val_pred = critic(planner_batch.obs)
        assert val_pred.shape == planner_batch.val.shape
        loss = ((val_pred - planner_batch.val)**2).mean()
        return loss
    
    critic, optimizer = nnx.merge(critic_graphdef, critic_state)
    loss, grads = nnx.value_and_grad(loss_fn)(critic)
    optimizer.update(grads)

    _, critic_state = nnx.split((critic, optimizer))
    
    target_critic_params = incremental_update(nnx.state(critic, nnx.Param), DV_state.target_critic_params, 1-config.policy_ema_rate)

    DV_state = DV_state._replace(critic_state=critic_state, rng=rng, target_critic_params=target_critic_params)

    return DV_state, loss
    

def policy_update(policy, DV_state, config, policy_batch):
    policy_model_state = DV_state.policy_model_state
    policy_model_graphdef = DV_state.policy_model_graphdef
    rng = DV_state.rng
    
    policy_td_obs, policy_td_next_obs, policy_td_act = policy_batch.obs[:,0,:], policy_batch.obs[:,1,:], policy_batch.act[:,0,:]
    condition = jnp.concatenate([policy_td_obs, policy_td_next_obs], axis=-1)

    key1, key2, rng = jax.random.split(rng, 3)
    xt, t, eps = policy.add_noise(policy_td_act, key1)
    def loss_fn(model):
        loss = (model(xt, t, condition, config.use_policy_condition, config.train_condition, key2) - eps) ** 2
        loss = loss * policy.loss_weight * (1 - policy.fix_mask)
        return loss.mean()
    
    policy_model, optimizer = nnx.merge(policy_model_graphdef, policy_model_state)
    loss, grads = nnx.value_and_grad(loss_fn)(policy_model)
    optimizer.update(grads)
    _, policy_model_state = nnx.split((policy_model, optimizer))
    target_policy_params = incremental_update(nnx.state(policy_model, nnx.Param), DV_state.target_policy_params, 1-config.policy_ema_rate)

    DV_state = DV_state._replace(policy_model_state=policy_model_state, rng=rng, target_policy_params=target_policy_params)
    return DV_state, loss


def planner_update(planner, DV_state, config, planner_batch, weighted_tensor=None):
    planner_model_state = DV_state.planner_model_state
    planner_model_graphdef = DV_state.planner_model_graphdef
    rng = DV_state.rng

    condition=None

    key1, key2, rng = jax.random.split(rng, 3)
    xt, t, eps = planner.add_noise(planner_batch.obs, key1)
    def loss_fn(model):
        loss = (model(xt, t, condition, config.use_planner_condition, config.train_condition, key2) - eps) ** 2
        loss = loss * planner.loss_weight * (1 - planner.fix_mask)
        if config.use_advantage_weighting:
            loss *= jnp.expand_dims(weighted_tensor, axis=1)
        return loss.mean()
    
    planner_model, optimizer = nnx.merge(planner_model_graphdef, planner_model_state)
    loss, grads = nnx.value_and_grad(loss_fn)(planner_model)
    optimizer.update(grads)
    _, planner_model_state = nnx.split((planner_model, optimizer))

    # ema
    target_planner_params = incremental_update(nnx.state(planner_model, nnx.Param), DV_state.target_planner_params, 1-config.planner_ema_rate)

    DV_state = DV_state._replace(planner_model_state=planner_model_state, rng=rng, target_planner_params=target_planner_params)

    return DV_state, loss

def update(planner, policy, DV_state, config, planner_dataset, policy_dataset):
    key1, key2, rng = jax.random.split(DV_state.rng, 3)
    DV_state = DV_state._replace(rng=rng)
    planner_batch = planner_dataset.sample(key1, config.batch_size)
    policy_batch = policy_dataset.sample(key2, config.batch_size)

    if config.use_advantage_weighting:
        weighted_tensor = jnp.exp((planner_batch.val - 1) * config.weight_factor)
    else: 
        weighted_tensor = None

    DV_state, planner_loss = planner_update(planner, DV_state, config, planner_batch, weighted_tensor=weighted_tensor)
    DV_state, critic_loss = critic_update(DV_state, config, planner_batch)
    DV_state, policy_loss = policy_update(policy, DV_state, config, policy_batch)

    return DV_state, DV_info(planner_loss=planner_loss, critic_loss=critic_loss, policy_loss=policy_loss)