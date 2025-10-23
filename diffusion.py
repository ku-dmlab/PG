import math
import jax
from flax import nnx
import jax.numpy as jnp
from typing import Optional
from network import BaseNNDiffusion
from copy import deepcopy
from typing import Optional, Union, Callable
import optax
from optax import incremental_update
from network import IdentityCondition
from util import SUPPORTED_NOISE_SCHEDULES, SUPPORTED_DISCRETIZATIONS, SUPPORTED_SAMPLING_STEP_SCHEDULE, at_least_ndim
from functools import partial


def epstheta_to_xtheta(x, alpha, sigma, eps_theta):
    """
    x_theta = (x - sigma * eps_theta) / alpha
    """
    return (x - sigma * eps_theta) / alpha


def xtheta_to_epstheta(x, alpha, sigma, x_theta):
    """
    eps_theta = (x - alpha * x_theta) / sigma
    """
    return (x - alpha * x_theta) / sigma



class ScoreDiffusion(nnx.Module):
    def __init__(self, nn_diffusion, nn_condition):
        self.diffusion = nn_diffusion
        self.condition = nn_condition
    
    def __call__(self, xt, t, condition, use_condition, train_condition,  rng):
        if use_condition:
            condition = self.condition(condition, rng, train_condition)
        output = self.diffusion(xt, t, condition, use_condition)
        return output


class ContinuousDiffusionSDE():
    def __init__(
            self,
            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition = None,
            # ----------------- Masks ----------------- #
            # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
            fix_mask = None,  # be in the shape of `x_shape`
            # Add loss weight
            loss_weight = None,  # be in the shape of `x_shape`
            # ------------------ Plugins ---------------- #
            # Add a classifier to enable classifier-guidance
            classifier = None,
            # ------------------ Training Params ---------------- #
            grad_clip_norm = None,
            ema_rate: float = 0.995,
            optim_params = None,
            # ------------------- Diffusion Params ------------------- #
            epsilon: float = 1e-3,
            noise_schedule = "cosine",
            noise_schedule_params = None,
            x_max = None,
            x_min = None,
            predict_noise: bool = True,
            planner_diffusion_gradient_steps = 1e6,
    ): 
        
        self.grad_clip_norm = grad_clip_norm
        self.ema_rate = ema_rate

        if nn_condition is None:
            nn_condition = IdentityCondition()

        self.model = ScoreDiffusion(nn_diffusion, nn_condition)

    
        self.classifier = classifier
        self.fix_mask = fix_mask[None, ] if fix_mask is not None else 0.
        self.loss_weight = loss_weight[None, ] if loss_weight is not None else 1.

        self.predict_noise = predict_noise
        self.epsilon = epsilon
        self.x_max = x_max
        self.x_min = x_min

        # ==================== Continuous Time-step Range ====================
        if noise_schedule == "cosine":
            self.t_diffusion = [epsilon, 0.9946]
        else:
            self.t_diffusion = [epsilon, 1.]

        # ===================== Noise Schedule ======================
        if isinstance(noise_schedule, str):
            if noise_schedule in SUPPORTED_NOISE_SCHEDULES.keys():
                self.noise_schedule_funcs = SUPPORTED_NOISE_SCHEDULES[noise_schedule]
                self.noise_schedule_params = noise_schedule_params
            else:
                raise ValueError(f"Noise schedule {noise_schedule} is not supported.")
        elif isinstance(noise_schedule, dict):
            self.noise_schedule_funcs = noise_schedule
            self.noise_schedule_params = noise_schedule_params
        else:
            raise ValueError("noise_schedule must be a callable or a string")
        
    
    def add_noise(self, x0, key=None, t=None, eps=None):
        key1, key2 = jax.random.split(key, 2)
        t = (jax.random.uniform(key=key1,  shape=(x0.shape[0],)) *
                (self.t_diffusion[1] - self.t_diffusion[0]) + self.t_diffusion[0]) if t is None else t

        eps = jax.random.normal(key=key2, shape=x0.shape) if eps is None else eps

        alpha, sigma = self.noise_schedule_funcs["forward"](t, **(self.noise_schedule_params or {}))
        alpha = at_least_ndim(alpha, x0.ndim)
        sigma = at_least_ndim(sigma, x0.ndim)

        xt = alpha * x0 + sigma * eps
        xt = (1. - self.fix_mask) * xt + self.fix_mask * x0

        return xt, t, eps
    
    # ==================== Sampling: Solving SDE/ODE ======================

    @partial(jax.jit, static_argnums=(0, 2, 4, 5, 6, 7, 8, 10))
    def sample(
            self,
            observation,
            model,
            rng,
            solver: str = "ddpm",
            planner_horizon: int = 4,
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform_continuous",
            temperature: float = 1.0,
            diffusion_x_sampling_steps: int = 0,
    ):
        prior = jnp.zeros((n_samples, planner_horizon - 1, observation.shape[-1]))
        prior = jnp.concatenate((jnp.expand_dims(observation, axis=1), prior), axis=1)

        xt = jax.random.normal(key=rng, shape=prior.shape) * temperature

        t_diffusion = self.t_diffusion

        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    t_diffusion, sample_steps)
                
        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {}))
        
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))

        def fn(xt, i):
            t = jnp.full((n_samples,), sample_step_schedule[i], dtype=jnp.float32)

            # guided sampling
            pred = model.diffusion(xt, t, condition=None, use_condition=False)

            # clip the prediction
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)

            if solver == "ddim":
                xt = (alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[i - 1] * eps_theta)
            else:
                NotImplementedError()

            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

            return xt, ()
        

        xt, () = jax.lax.scan(fn, xt, jnp.flip(jnp.array(loop_steps)))
           
        if self.clip_pred():
            xt = xt.clip(self.x_min, self.x_max)

        return xt


    @partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 10))
    def sample_prior(
            self,
            observation,
            prior,
            rng,
            solver: str = "ddpm",
            planner_horizon: int = 4,
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform_continuous",
            temperature: float = 1.0,
            diffusion_x_sampling_steps: int = 0,
    ):
        
        batch_size = observation.shape[0]
        prior = jnp.concatenate((jnp.expand_dims(observation, axis=1), prior), axis=1)
        xt = deepcopy(prior)

        t_diffusion = self.t_diffusion

        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    t_diffusion, sample_steps)
                
        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {}))
        
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))

        def fn(xt, i):
            t = jnp.full((batch_size,), sample_step_schedule[i], dtype=jnp.float32)
            pred = self.model.diffusion(xt, t, condition=None, use_condition=False)
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)

            if solver == "ddim":
                xt = (alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[i - 1] * eps_theta)
            else:
                NotImplementedError()

            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            return xt, ()
        

        xt, () = jax.lax.scan(fn, xt, jnp.flip(jnp.array(loop_steps)))
           
        if self.clip_pred(): 
            xt = xt.clip(self.x_min, self.x_max)

        return xt
    

    def sample_prior_train(
            self,
            observation,
            prior,
            config,
            rng
    ):
        solver = config.planner_solver
        sample_steps=config.planner_sampling_steps_train
        sample_step_schedule = "uniform_continuous"
        diffusion_x_sampling_steps = 0


        batch_size = observation.shape[0]
        prior = jnp.concatenate((jnp.expand_dims(observation, axis=1), prior), axis=1)
        xt = deepcopy(prior)


        t_diffusion = self.t_diffusion

        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    t_diffusion, sample_steps)
                
        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {}))
        
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))

        def fn(xt, i):
            t = jnp.full((batch_size,), sample_step_schedule[i], dtype=jnp.float32)
            pred = self.model.diffusion(xt, t, condition=None, use_condition=False)
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)

            if solver == "ddim":
                xt = (alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[i - 1] * eps_theta)
            else:
                NotImplementedError()

            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            return xt, ()
        

        xt, () = jax.lax.scan(fn, xt, jnp.flip(jnp.array(loop_steps)))
           
        if self.clip_pred(): 
            xt = xt.clip(self.x_min, self.x_max)

        return xt
    
    
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    def clip_prediction(self, pred, xt, alpha, sigma):
        """
        Clip the prediction at each sampling step to stablize the generation.
        (xt - alpha * x_max) / sigma <= eps <= (xt - alpha * x_min) / sigma
                               x_min <= x0  <= x_max
        """
        if self.predict_noise:
            if self.clip_pred():
                upper_bound = (xt - alpha * self.x_min) / sigma if self.x_min is not None else None
                lower_bound = (xt - alpha * self.x_max) / sigma if self.x_max is not None else None
                pred = pred.clip(lower_bound, upper_bound)
        else:
            if self.clip_pred():
                pred = pred.clip(self.x_min, self.x_max)

        return pred
    

    

class DiscreteDiffusionSDE():
    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition = None,

            # ----------------- Masks ----------------- #
            # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
            fix_mask = None,  # be in the shape of `x_shape`
            # Add loss weight
            loss_weight= None,  # be in the shape of `x_shape`

            # ------------------ Plugins ---------------- #
            # Add a classifier to enable classifier-guidance
            classifier = None,

            # ------------------ Training Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,

            # ------------------- Diffusion Params ------------------- #
            epsilon: float = 1e-3,

            diffusion_steps: int = 1000,
            discretization: Union[str, Callable] = "uniform",

            noise_schedule = "cosine",
            noise_schedule_params: Optional[dict] = None,

            x_max = None,
            x_min= None,

            predict_noise: bool = True
    ):
        
        self.grad_clip_norm = grad_clip_norm
        self.ema_rate = ema_rate

        if nn_condition is None:
            nn_condition = IdentityCondition()

        self.model = ScoreDiffusion(nn_diffusion, nn_condition)

        self.classifier = classifier

        self.fix_mask = fix_mask[None, ] if fix_mask is not None else 0.
        self.loss_weight = loss_weight[None, ] if loss_weight is not None else 1.

        self.predict_noise = predict_noise
        self.epsilon = epsilon
        self.x_max = x_max
        self.x_min = x_min

        self.diffusion_steps = diffusion_steps

        if 1. / diffusion_steps < epsilon:
            raise ValueError("epsilon is too large for the number of diffusion steps")

        # ================= Discretization =================
        if isinstance(discretization, str):
            if discretization in SUPPORTED_DISCRETIZATIONS.keys():
                self.t_diffusion = SUPPORTED_DISCRETIZATIONS[discretization](diffusion_steps, epsilon)
            else:
                Warning(f"Discretization method {discretization} is not supported. "
                        f"Using uniform discretization instead.")
                self.t_diffusion = SUPPORTED_DISCRETIZATIONS["uniform"](diffusion_steps, epsilon)
        elif callable(discretization):
            self.t_diffusion = discretization(diffusion_steps, epsilon)
        else:
            raise ValueError("discretization must be a callable or a string")

        # ================= Noise Schedule =================
        if isinstance(noise_schedule, str):
            if noise_schedule in SUPPORTED_NOISE_SCHEDULES.keys():
                self.alpha, self.sigma = SUPPORTED_NOISE_SCHEDULES[noise_schedule]["forward"](
                    self.t_diffusion, **(noise_schedule_params or {}))
            else:
                raise ValueError(f"Noise schedule {noise_schedule} is not supported.")
        elif isinstance(noise_schedule, dict):
            self.alpha, self.sigma = noise_schedule["forward"](self.t_diffusion, **(noise_schedule_params or {}))
        else:
            raise ValueError("noise_schedule must be a callable or a string")

        self.logSNR = jnp.log(self.alpha / self.sigma)


    # ==================== Training: Score Matching ======================
    def add_noise(self, x0, key=None, t=None, eps=None):
        key1, key2 = jax.random.split(key, 2)
        t = jax.random.randint(key1, (x0.shape[0],), 0, self.diffusion_steps) if t is None else t
        eps = jax.random.normal(key2, shape=x0.shape) if eps is None else eps

        alpha = at_least_ndim(self.alpha[t], x0.ndim)
        sigma = at_least_ndim(self.sigma[t], x0.ndim)

        xt = alpha * x0 + sigma * eps
        xt = (1. - self.fix_mask) * xt + self.fix_mask * x0

        return xt, t, eps

    @partial(jax.jit, static_argnums=(0, 2, 4, 5, 6, 7, 10, 11))
    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior,
            model,
            rng,
            # ----------------- sampling ----------------- #
            solver: str = "ddpm",
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform",
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            # ----------- Diffusion-X sampling ----------
            diffusion_x_sampling_steps: int = 0,
    ):
        key1, key2, rng = jax.random.split(rng, 3)

        diffusion_steps = self.diffusion_steps
        xt = jax.random.normal(key=key1, shape=prior.shape) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

        condition_vec_cfg = model.condition(condition = condition_cfg, mask=mask_cfg, key=key2, train=False) if condition_cfg is not None else None

        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    diffusion_steps, sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
            
        alphas = self.alpha[sample_step_schedule]
        sigmas = self.sigma[sample_step_schedule]
        logSNRs = jnp.log(alphas / sigmas)
        hs = jnp.concatenate((jnp.array([0]), logSNRs[:-1] - logSNRs[1:]), axis=0)  
        stds_1 = sigmas[:-1] / sigmas[1:] * jnp.sqrt(1 - (alphas[1:] / alphas[:-1]) ** 2)
        stds = jnp.concatenate((jnp.array([0]), stds_1), axis=0)

        buffer = []

        # ===================== Denoising Loop ========================
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))

        def fn(input_tuple, i):
            xt, rng = input_tuple
            t = jnp.full((n_samples,), sample_step_schedule[i], dtype=jnp.int32)
            pred = model.diffusion(xt, t, condition_vec_cfg)

            # clip the prediction
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            # noise & data prediction
            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)

            key, rng = jax.random.split(rng, 2)
            # one-step update
            if solver == "ddpm":
                xt = (
                        (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta) +
                        jnp.sqrt(sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8) * eps_theta)
                xt += (stds[i] * jax.random.normal(key=key, shape=xt.shape)) * (i > 1)
            else:
                NotImplementedError()
            
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

            return (xt, rng), ()        

        input_tuple, () = jax.lax.scan(fn, (xt, rng), jnp.flip(jnp.array(loop_steps)))
        xt, rng = input_tuple
        
        if self.clip_pred():
            xt = xt.clip(self.x_min, self.x_max)
        
        return xt

    
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)
    
    def clip_prediction(self, pred, xt, alpha, sigma):
        """
        Clip the prediction at each sampling step to stablize the generation.
        (xt - alpha * x_max) / sigma <= eps <= (xt - alpha * x_min) / sigma
                               x_min <= x0  <= x_max
        """
        if self.predict_noise:
            if self.clip_pred():
                upper_bound = (xt - alpha * self.x_min) / sigma if self.x_min is not None else None
                lower_bound = (xt - alpha * self.x_max) / sigma if self.x_max is not None else None
                pred = pred.clip(lower_bound, upper_bound)
        else:
            if self.clip_pred():
                pred = pred.clip(self.x_min, self.x_max)

        return pred



    def guided_sampling(
        self, xt, t, alpha, sigma,
        model,
        condition_cfg=None, w_cfg: float = 0.0,
        condition_cg=None, w_cg: float = 0.0,
        requires_grad: bool = False):
        """
        One-step epsilon/x0 prediction with guidance.
        """

        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad)

        return pred


    # ==================== Sampling: Solving SDE/ODE ======================

    def classifier_guidance(
            self, xt, t, alpha, sigma,
            model, condition=None, w: float = 1.0,
            pred=None):
        """
        Guided Sampling CG:
        bar_eps = eps - w * sigma * grad
        bar_x0  = x0 + w * (sigma ** 2) * alpha * grad
        """
        if pred is None:
            pred = model["diffusion"](xt, t, None)
        if self.classifier is None or w == 0.0:
            return pred, None
        else:
            log_p, grad = self.classifier.gradients(xt.clone(), t, condition)
            if self.predict_noise:
                pred = pred - w * sigma * grad
            else:
                pred = pred + w * ((sigma ** 2) / alpha) * grad

        return pred, log_p


    def classifier_free_guidance(
            self, xt, t,
            model, condition=None, w: float = 1.0,
            pred=None, pred_uncond=None,
            requires_grad: bool = False):
        """
        Guided Sampling CFG:
        bar_eps = w * pred + (1 - w) * pred_uncond
        bar_x0  = w * pred + (1 - w) * pred_uncond
        """

        pred = model["diffusion"](xt, t, condition)
        pred_uncond = 0.

        return pred