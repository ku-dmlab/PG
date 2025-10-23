import math
import os
import random
from typing import Union, Dict, Callable
import jax
import numpy as np
import jax.numpy as jnp
from flax import nnx


class SinusoidalEmbedding(nnx.Module):
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = jnp.einsum('...i,j->...ij', x, emb)
        # emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), -1)
        return emb


# ================= Noise schedules =================
def linear_noise_schedule(
    t_diffusion, beta0: float = 0.1, beta1: float = 20.0
):
    log_alpha = -(beta1 - beta0) / 4.0 * (t_diffusion**2) - beta0 / 2.0 * t_diffusion
    alpha = jnp.exp(log_alpha)
    sigma = jnp.sqrt(1.0 - alpha**2)
    return alpha, sigma


def inverse_linear_noise_schedule(
    alpha,
    sigma,
    logSNR,
    beta0: float = 0.1,
    beta1: float = 20.0,
):
    assert (logSNR is not None) or (alpha is not None and sigma is not None)
    lmbda = jnp.log(alpha / sigma) if logSNR is None else logSNR
    t_diffusion = (2 * jnp.log(1 + jnp.exp(-2 * lmbda)) /
                   (beta0 + (beta0**2 + 2 * (beta1 - beta0) * jnp.log(1 + jnp.exp(-2 * lmbda)))))
    return t_diffusion


def cosine_noise_schedule(t_diffusion, s: float = 0.008):
    t_diffusion = t_diffusion.at[-1].set(0.9946)
    alpha = jnp.cos(jnp.pi / 2.0 * (t_diffusion + s) / (1 + s)) / jnp.cos(jnp.pi / 2.0 * s / (1 + s))
    sigma = jnp.sqrt(1.0 - alpha**2)
    return alpha, sigma


def inverse_cosine_noise_schedule(
    alpha = None,
    sigma = None,
    logSNR = None,
    s: float = 0.008,
):
    assert (logSNR is not None) or (alpha is not None and sigma is not None)
    lmbda = jnp.log(alpha / sigma) if logSNR is None else logSNR
    t_diffusion = (
        2 * (1 + s) / jnp.pi * jnp.arccos(jnp.exp(
            -0.5 * jnp.log(1 + jnp.exp(-2 * lmbda))
            + jnp.log(np.cos(jnp.pi * s / 2 / (s + 1))))) - s)
    return t_diffusion


SUPPORTED_NOISE_SCHEDULES = {
    "linear": {
        "forward": linear_noise_schedule,
        "reverse": inverse_linear_noise_schedule,
    },
    "cosine": {
        "forward": cosine_noise_schedule,
        "reverse": inverse_cosine_noise_schedule,
    },
}

def uniform_discretization(T: int = 1000, eps: float = 1e-3):
    return jnp.linspace(eps, 1.0, T)


SUPPORTED_DISCRETIZATIONS = {
    "uniform": uniform_discretization,
}

def uniform_sampling_step_schedule(T: int = 1000, sampling_steps: int = 10):
    return jnp.linspace(0, T - 1, sampling_steps + 1, dtype=jnp.int32)

def uniform_sampling_step_schedule_continuous(trange=None, sampling_steps: int = 10):
    if trange is None:
        trange = [1e-3, 1.0]
    return jnp.linspace(trange[0], trange[1], sampling_steps + 1, dtype=jnp.float32)


SUPPORTED_SAMPLING_STEP_SCHEDULE = {
    "uniform": uniform_sampling_step_schedule,
    "uniform_continuous": uniform_sampling_step_schedule_continuous,
}




def at_least_ndim(x, ndim: int, pad: int = 0):
    """ Add dimensions to the input tensor to make it at least ndim-dimensional.

    Args:
        x: Union[np.ndarray, torch.Tensor, int, float], input tensor
        ndim: int, minimum number of dimensions
        pad: int, padding direction. `0`: pad in the last dimension, `1`: pad in the first dimension

    Returns:
        Any of these 2 options

        - np.ndarray or torch.Tensor: reshaped tensor
        - int or float: input value

    Examples:
        >>> x = np.random.rand(3, 4)
        >>> at_least_ndim(x, 3, 0).shape
        (3, 4, 1)
        >>> x = torch.randn(3, 4)
        >>> at_least_ndim(x, 4, 1).shape
        (1, 1, 3, 4)
        >>> x = 1
        >>> at_least_ndim(x, 3)
        1
    """
    if isinstance(x, np.ndarray):
        if ndim > x.ndim:
            if pad == 0:
                return np.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return np.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, jax.Array):
        if ndim > x.ndim:
            if pad == 0:
                return jnp.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return jnp.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, (int, float)):
        return x
    else:
        raise ValueError(f"Unsupported type {type(x)}")



class GaussianNormalizer():

    def __init__(self, X, start_dim: int = -1):
        total_dims = X.ndim
        if start_dim < 0:
            start_dim = total_dims + start_dim

        axes = tuple(range(start_dim))

        self.mean = jnp.mean(X, axis=axes)
        self.std = jnp.std(X, axis=axes)
        self.std = self.std.at[self.std == 0].set(1.)

    def normalize(self, x):
        ndim = x.ndim
        return (x - at_least_ndim(self.mean, ndim, 1)) / at_least_ndim(self.std, ndim, 1)

    def unnormalize(self, x):
        ndim = x.ndim
        return x * at_least_ndim(self.std, ndim, 1) + at_least_ndim(self.mean, ndim, 1)