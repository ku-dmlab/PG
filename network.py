from flax import nnx
import jax.numpy as jnp
import jax 
import math
import jax
from flax import nnx
import jax.numpy as jnp
from typing import Optional
from util import at_least_ndim, SinusoidalEmbedding
from functools import partial
import tensorflow_probability.substrates.jax as tfp
import copy

tfd = tfp.distributions
tfb = tfp.bijectors


class PositionalEmbedding(nnx.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False, rngs: nnx.Rngs=nnx.Rngs(0)):
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def __call__(self, x):
        freqs = jnp.arange(start=0, stop=self.dim // 2, dtype=jnp.float32)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = jnp.outer(x, freqs.astype(x.dtype))
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=1)
        return x


class FourierEmbedding(nnx.Module):
    def __init__(self, dim: int, scale=16, rngs:nnx.Rngs=nnx.Rngs(0)):
        key = rngs.key()
        self.freqs = nnx.Param(jax.random.normal(key=key, shape=(dim // 8)) * scale, requires_grad=False)
        # self.freqs = nn.Parameter(torch.randn(dim // 8) * scale, requires_grad=False)
        self.mlp = nnx.Sequential(
            nnx.Linear(dim // 4, dim, rngs=rngs), jax.nn.mish, nnx.Linear(dim, dim, rngs=rngs)
        )

    def __call__(self, x):
        emb = jnp.einsum('...i,j->...ij', x, (2 * jnp.pi * self.freqs))
        # emb = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        emb = jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], -1)
        return self.mlp(emb)
           

SUPPORTED_TIMESTEP_EMBEDDING = {
    "positional": PositionalEmbedding,
    "fourier": FourierEmbedding
}


class TransformerBlock(nnx.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0, norm_type="post", rngs: nnx.Rngs=nnx.Rngs(0)):
        # self.norm_type = norm_type
        
        self.norm1 = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(n_heads, hidden_size,
                                           kernel_init=nnx.initializers.xavier_uniform(),
                                           bias_init=nnx.initializers.constant(0),
                                           dropout_rate=dropout, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)

        def approx_gelu(x): return nnx.gelu(x, approximate=True)

        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size, hidden_size * 4,
                       kernel_init=nnx.initializers.xavier_uniform(),
                       bias_init=nnx.initializers.constant(0),
                       rngs=rngs), 
                       approx_gelu, 
                       nnx.Dropout(dropout, rngs=rngs),
            nnx.Linear(hidden_size * 4, hidden_size, 
                       kernel_init=nnx.initializers.xavier_uniform(),
                       bias_init=nnx.initializers.constant(0),
                       rngs=rngs))

    def __call__(self, x):
        # if norm_type == "post":
        #     x = self.norm1(x + self.attn(x, x, x)[0])
        #     x = self.norm2(x + self.mlp(x))
        # elif norm_type == "pre":
        x = self.norm1(x)
        x = x + self.attn(x, x, x, decode=False)
        x = x + self.mlp(self.norm2(x))
        # else:
        #     raise NotImplementedError
        return x

class DAHorizonCritic(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        norm_type: str = "post",
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.in_dim, self.emb_dim = in_dim, emb_dim
        self.d_model = d_model

        self.x_proj = nnx.Linear(in_dim, d_model, 
                                 kernel_init=nnx.initializers.xavier_uniform(),
                                 bias_init=nnx.initializers.constant(0),
                                 rngs=rngs)

        self.pos_emb = SinusoidalEmbedding(d_model)
        # self.pos_emb_cache = None

        self.blocks = [TransformerBlock(d_model, n_heads, dropout, norm_type, rngs=rngs) for _ in range(depth)]
        self.final_layer = nnx.Linear(d_model, 1,
                                      kernel_init=nnx.initializers.xavier_uniform(),
                                      bias_init=nnx.initializers.constant(0),
                                      rngs=rngs)
        
    def __call__(self, x):
        """
        Input:
            x:          (b, horizon, in_dim)

        Output:
            y:          (b, horizon, in_dim)
        """
        # a= jnp.expand_dims(jnp.repeat(jnp.expand_dims(x[0, 0, :], axis=0), repeats=128, axis=0), axis=1)
        # x = jnp.concatenate((a, x[:, 1:, :]), axis=1)

        # if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
        pos_emb_cache = self.pos_emb(jnp.arange(x.shape[1]))

        x = self.x_proj(x) + jnp.expand_dims(pos_emb_cache, axis=0)

        for block in self.blocks:
            x = block(x)
        x = self.final_layer(x)

        x = x[:, 0, :]
        
        return x
    
    @partial(jax.jit, static_argnums=(0))
    def eval_forward(self, x):
        return self.__call__(x)



class BaseNNDiffusion(nnx.Module):
    """
    The neural network backbone for the Diffusion model used for score matching
     (or training a noise predictor) should take in three inputs.
     The first input is the noisy data.
     The second input is the denoising time step, which can be either as a discrete variable
     or a continuous variable, specified by the parameter `discrete_t`.
     The third input is the condition embedding that has been processed through the `nn_condition`.
     In the general case, we assume that there may be multiple conditions,
     which are inputted as a tensor dictionary, or a single condition, directly inputted as a tensor.
    """

    def __init__(
        self, emb_dim: int, 
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
        rngs: nnx.Rngs=nnx.Rngs(0)
    ):
        assert timestep_emb_type in SUPPORTED_TIMESTEP_EMBEDDING.keys()
        timestep_emb_params = timestep_emb_params or {}
        self.map_noise = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](emb_dim, rngs=rngs, **timestep_emb_params)

    def __call__(self, x, noise, condition = None):
        """
        Input:
            x:          (b, horizon, in_dim)
            noise:      (b, )
            condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

        Output:
            y:          (b, horizon, in_dim)
        """
        raise NotImplementedError



class DAMlp(BaseNNDiffusion):
    def __init__(
        self, 
        obs_dim: int,
        act_dim: int,
        emb_dim: int = 16, 
        hidden_dim: int = 256,
        timestep_emb_type: str = "positional",
        timestep_emb_params= None,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params, rngs=rngs)
        
        self.time_mlp = nnx.Sequential(
            nnx.Linear(emb_dim, emb_dim * 2, rngs=rngs), 
            jax.nn.mish, 
            nnx.Linear(emb_dim * 2, emb_dim, rngs=rngs))
        
        self.mid_layer = nnx.Sequential(
            nnx.Linear(obs_dim * 2 + act_dim + emb_dim, hidden_dim, rngs=rngs), jax.nn.mish,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs), jax.nn.mish,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs), jax.nn.mish)
        
        self.final_layer = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        
    def __call__(self, x, noise, condition = None, use_condition = None):
        """
        Input:
            x:          (b, act_dim)
            noise:      (b, )
            condition:  (b, obs_dim * 2)

        Output:
            y:          (b, act_dim)
        """
        t = self.time_mlp(self.map_noise(noise))
        x = jnp.concatenate([x, t, condition], -1)
        x = self.mid_layer(x)
        
        return self.final_layer(x)

def get_mask(mask, mask_shape: tuple, dropout: float, train: bool, key):
    if train:
        mask = (jax.random.uniform(key, mask_shape) > dropout).astype(jnp.float32)
    else:
        mask = 1. if mask is None else mask
    return mask


class IdentityCondition(nnx.Module):
    """
    Identity condition does not change the input condition.

    Input:
        - condition: (b, *cond_in_shape)
        - mask :     (b, ) or None, None means no mask

    Output:
        - condition: (b, *cond_in_shape)
    """

    def __init__(self, dropout: float = 0.25):
        self.dropout = dropout

    def __call__(self, condition, key, train=True, mask = None):
        mask = at_least_ndim(get_mask(
            mask, (condition.shape[0],), self.dropout, train, key), condition.ndim)
        return condition * mask





def modulate(x, shift, scale):
    return x * (1 + jnp.expand_dims(scale, axis=1)) + jnp.expand_dims(shift, axis=1)

class DiTBlock(nnx.Module):
    """ A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning. """

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0, rngs: nnx.Rngs = nnx.Rngs(0)):
        self.norm1 = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(n_heads, hidden_size,
                                           kernel_init=nnx.initializers.xavier_uniform(),
                                           bias_init=nnx.initializers.constant(0),
                                           dropout_rate=dropout, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)

        def approx_gelu(x): return nnx.gelu(x, approximate=True)

        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size, 
                       hidden_size * 4, 
                       kernel_init=nnx.initializers.xavier_uniform(),
                       bias_init=nnx.initializers.constant(0),
                       rngs=rngs), 
            approx_gelu, 
            nnx.Dropout(dropout, rngs=rngs),
            nnx.Linear(hidden_size * 4, hidden_size, 
                        kernel_init=nnx.initializers.xavier_uniform(),
                        bias_init=nnx.initializers.constant(0),
                       rngs=rngs))
        
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu, 
            nnx.Linear(hidden_size, 
                       hidden_size * 6, 
                       kernel_init=nnx.initializers.constant(0),
                       bias_init=nnx.initializers.constant(0),
                       rngs=rngs))

    def __call__(self, x, t):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(self.adaLN_modulation(t), 6, axis=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        # Attention layer dimensions might not match
        x = x + jnp.expand_dims(gate_msa, axis=1) * self.attn(x, x, x, decode=False)
        x = x + jnp.expand_dims(gate_mlp, axis=1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer1d(nnx.Module):
    def __init__(self, hidden_size: int, out_dim: int, rngs: nnx.Rngs=nnx.Rngs(0)):
        self.norm_final = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.linear = nnx.Linear(hidden_size, out_dim, 
                                 kernel_init=nnx.initializers.constant(0),
                                 bias_init=nnx.initializers.constant(0),
                                 rngs=rngs)
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu, 
            nnx.Linear(hidden_size, 
                       2 * hidden_size,
                       kernel_init=nnx.initializers.constant(0),
                       bias_init=nnx.initializers.constant(0),   
                       rngs=rngs))

    def __call__(self, x, t):
        shift, scale = jnp.split(self.adaLN_modulation(t), 2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT1d(BaseNNDiffusion):
    def __init__(
        self,
        in_dim: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params, rngs=rngs)
        self.in_dim, self.emb_dim = in_dim, emb_dim
        self.d_model = d_model

        self.x_proj = nnx.Linear(
            in_dim, d_model,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.constant(0),
            rngs=rngs
        )
        self.map_emb = nnx.Sequential(
            nnx.Linear(
                emb_dim, d_model,
                kernel_init=nnx.initializers.normal(stddev=0.02),
                bias_init=nnx.initializers.constant(0),
                rngs=rngs
            ),
            jax.nn.mish,
            nnx.Linear(
                d_model, d_model,
                kernel_init=nnx.initializers.normal(stddev=0.02),
                bias_init=nnx.initializers.constant(0),
                rngs=rngs
            ),
            jax.nn.mish
        )

        self.pos_emb = SinusoidalEmbedding(d_model)
        # self.pos_emb_cache = None

        # Replace module list with regular list
        self.blocks = [DiTBlock(d_model, n_heads, dropout, rngs=rngs) for _ in range(depth)]
        self.final_layer = FinalLayer1d(d_model, in_dim, rngs=rngs)
        

    def __call__(self, x, noise, condition, use_condition):
        """
        Input:
            x:          (b, horizon, in_dim)
            noise:      (b, )
            condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

        Output:
            y:          (b, horizon, in_dim)
        """
        # if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
        pos_emb_cache = self.pos_emb(jnp.arange(x.shape[1]))

        x = self.x_proj(x) + jnp.expand_dims(pos_emb_cache, axis=0)
        emb = self.map_noise(noise)
        if use_condition:
            emb = emb + condition
        emb = self.map_emb(emb)

        for block in self.blocks:
            x = block(x, emb)
        x = self.final_layer(x, emb)
        return x




class MLP(nnx.Module):
    def __init__(self, din, dout = 1, hidden_dims = [256, 256], activation = nnx.relu, rngs: nnx.Rngs = nnx.Rngs(0), activate_final: bool = False, dropout_rate: float = 0.0, layer_norm: bool = False):
        dims = [din] + hidden_dims + [dout]
        
        layer = []
        for i in range(len(dims) - 1):
            layer.append(nnx.Linear(dims[i], dims[i+1], rngs=rngs, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))))
            if i < len(dims) - 2:
                if dropout_rate > 0:
                    layer.append(nnx.Dropout(dropout_rate, rngs=rngs))
                layer.append(activation)
                if layer_norm:
                    layer.append(nnx.LayerNorm(dims[i+1], rngs=rngs))
                
        if activate_final:
            layer.append(activation)
        
        self.layer = nnx.Sequential(
            *layer
        )
    
    def __call__(self, x):
        return self.layer(x)
    

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nnx.Module):
    def __init__(self,
                 observation_dim,
                 action_dim,
                 hidden_dims = [256, 256],
                 activation = nnx.relu,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 tanh_squash: bool = False,
                 temperature = 1.0,
                 log_std_scale: float = 1e-3,
                 layer_norm: bool = False,
                 ):
        
        self.temparature = temperature
        self.mlp_layer = MLP(observation_dim, hidden_dims[-1], hidden_dims[:-1], activation=activation, rngs = rngs, activate_final=True, layer_norm=layer_norm)
        self.mean_layer = nnx.Linear(
            hidden_dims[-1], action_dim, rngs = rngs, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))
        )
        self.std_layer = nnx.Linear(
            hidden_dims[-1], action_dim, rngs = rngs, kernel_init=nnx.initializers.orthogonal(log_std_scale)
        )
        self.tanh_squash = tanh_squash
        self.action_dim = action_dim

    def __call__(self, observations):
        x = self.mlp_layer(observations)
        
        means = self.mean_layer(x)
        if not self.tanh_squash:
            means = jnp.tanh(means)
        log_stds = self.std_layer(x)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        
        dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) * self.temparature)
        
        if self.tanh_squash:
            return tfd.TransformedDistribution(distribution=dist,
                                               bijector=tfb.Tanh())
            
        else:
            return dist

class TanhDeterministic(nnx.Module):
    def __init__(self, observation_dim, action_dim, hidden_dims = [256, 256], squash_mean:float=1.0, rngs: nnx.Rngs = nnx.Rngs(0)):
        self.mlp_layer = MLP(observation_dim, action_dim, hidden_dims, activation=nnx.relu, rngs = rngs, activate_final=False, layer_norm=True)
        self.squash_mean = squash_mean
    
    def __call__(self, observations):
        return jnp.tanh(self.mlp_layer(observations)) * self.squash_mean
    
    @partial(jax.jit, static_argnums=(0))
    def eval_forward(self, observations):
        return self.__call__(observations)
    


class V(nnx.Module):
    def __init__(self, obs_dim, hidden_dim: int = 256, rngs:nnx.Rngs = nnx.Rngs(0)):
        self.V = nnx.Sequential(
            nnx.Linear(obs_dim, hidden_dim, rngs=rngs),
            nnx.LayerNorm(hidden_dim, rngs=rngs), jax.nn.mish,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.LayerNorm(hidden_dim, rngs=rngs), jax.nn.mish,
            nnx.Linear(hidden_dim, 1, rngs=rngs))

    def __call__(self, obs):
        return self.V(obs)
    
    @partial(jax.jit, static_argnums=(0))
    def eval_forward(self, x):
        return self.__call__(x)
    
    
class TanhStochasticGRU(nnx.Module):
    def __init__(
        self,
        observation_dim,
        planner_horizon: int = 4,
        hidden_dim: int = 256,
        squash_mean: float = 1.0,
        divergence: str = 'kl',
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        # Shared feature extractor
        self.linear_1 = nnx.Linear(observation_dim, hidden_dim, rngs=rngs)
        self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.gru = nnx.GRUCell(in_features=hidden_dim, hidden_features=hidden_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        # Policy heads: one for mean, one for log_std
        self.linear_mean = nnx.Linear(hidden_dim, observation_dim, rngs=rngs)
        self.linear_log_std = nnx.Linear(hidden_dim, observation_dim, rngs=rngs)
        self.squash_mean = squash_mean
        self.planner_horizon = planner_horizon
        self.divergence = divergence

    def __call__(self, x):
        initial_rnn_state = self.gru.initialize_carry(x.shape)
        x = self.linear_1(x)
        x = self.ln1(x)
        x = nnx.relu(x)
        carry = (x, initial_rnn_state)

        def step_fn(carry, _):
            x, rnn_state = carry
            x, rnn_state = self.gru(rnn_state, x)
            x = self.ln2(x)
            x = nnx.relu(x)
            # Policy heads: mean and log_std calculation
            mean = self.linear_mean(x)
            log_std = self.linear_log_std(x)
            dummy_x = jnp.zeros(shape=x.shape)
        
            return (dummy_x, rnn_state), (mean, log_std)

        (dummy_x, hidden_states), (means, log_stds) = jax.lax.scan(step_fn, carry, None, length=self.planner_horizon - 1)
        
        means = jnp.transpose(means, axes=(1, 0, 2))
        log_stds = jnp.transpose(log_stds, axes=(1, 0, 2))
        if self.divergence == 'pearson_chi2':
            log_stds = -jax.nn.softplus(log_stds) - 0.5
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        
        return dist
    
    @partial(jax.jit, static_argnums=(0))
    def eval_forward(self, x):
        return self.__call__(x)



class TanhDeterministicGRU(nnx.Module):
    def __init__(
        self,
        observation_dim,
        planner_horizon: int = 4,
        hidden_dim: int = 256,
        squash_mean: float = 1.0,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        # Shared feature extractor
        self.linear_1 = nnx.Linear(observation_dim, hidden_dim, rngs=rngs)
        self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.gru = nnx.GRUCell(in_features=hidden_dim, hidden_features=hidden_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        # Policy heads: one for mean, one for log_std
        self.linear_mean = nnx.Linear(hidden_dim, observation_dim, rngs=rngs)
        # self.linear_log_std = nnx.Linear(hidden_dim, observation_dim, rngs=rngs)
        self.squash_mean = squash_mean
        self.planner_horizon = planner_horizon

    def __call__(self, x):
        initial_rnn_state = self.gru.initialize_carry(x.shape)
        x = self.linear_1(x)
        x = self.ln1(x)
        x = nnx.relu(x)
        carry = (x, initial_rnn_state)

        def step_fn(carry, _):
            x, rnn_state = carry
            x, rnn_state = self.gru(rnn_state, x)
            x = self.ln2(x)
            x = nnx.relu(x)
            # Policy heads: mean and log_std calculation
            mean = self.linear_mean(x)
            # log_std = self.linear_log_std(x)
            dummy_x = jnp.zeros(shape=x.shape)
        
            return (dummy_x, rnn_state), (mean)

        (dummy_x, hidden_states), (means) = jax.lax.scan(step_fn, carry, None, length=self.planner_horizon - 1)
        
        means = jnp.transpose(means, axes=(1, 0, 2))

        return means
    
    @partial(jax.jit, static_argnums=(0))
    def eval_forward(self, x):
        return self.__call__(x)






class MixtureStochasticGRU(nnx.Module):
    def __init__(
        self,
        observation_dim,
        planner_horizon: int = 4,
        hidden_dim: int = 256,
        num_components: int = 5,
        squash_mean: float = 1.0,
        divergence: str = 'kl',
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        # Shared feature extractor
        self.linear_1 = nnx.Linear(observation_dim, hidden_dim, rngs=rngs)
        self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.gru = nnx.GRUCell(in_features=hidden_dim, hidden_features=hidden_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        # Mixture components
        self.num_components = num_components
        self.linear_mix_logits = nnx.Linear(hidden_dim, num_components, rngs=rngs)
        self.linear_mean = nnx.Linear(hidden_dim, observation_dim * num_components, rngs=rngs)
        self.linear_log_std = nnx.Linear(hidden_dim, observation_dim * num_components, rngs=rngs)
        
        self.squash_mean = squash_mean
        self.planner_horizon = planner_horizon
        self.divergence = divergence

    def __call__(self, x):
        batch_size = x.shape[0]
        obs_dim = x.shape[-1]
        # Initial state
        h0 = self.gru.initialize_carry(x.shape)
        # Embed observations
        x = self.linear_1(x)
        x = self.ln1(x)
        x = nnx.relu(x)
        carry = (x, h0)

        def step_fn(carry, _):
            x, h = carry
            x, h = self.gru(h, x)
            x = self.ln2(x)
            x = nnx.relu(x)
            # Mixture heads
            logits = self.linear_mix_logits(x)
            means = self.linear_mean(x)
            log_stds = self.linear_log_std(x)
            dummy = jnp.zeros_like(x)
            return (dummy, h), (means, log_stds, logits)

        (_, _), (means_seq, log_stds_seq, logits_seq) = \
            jax.lax.scan(step_fn, carry, None, length=self.planner_horizon-1)

        # [steps, batch, ...] -> [batch, steps, ...]
        means_seq = jnp.transpose(means_seq, (1, 0, 2))
        log_stds_seq = jnp.transpose(log_stds_seq, (1, 0, 2))
        logits_seq = jnp.transpose(logits_seq, (1, 0, 2))

        # Reshape to mixture components
        means_seq = means_seq.reshape(batch_size, self.planner_horizon-1, self.num_components, obs_dim)
        log_stds_seq = log_stds_seq.reshape(batch_size, self.planner_horizon-1, self.num_components, obs_dim)

        # Apply divergence-specific transform
        if self.divergence == 'pearson_chi2':
            log_stds_seq = -jax.nn.softplus(log_stds_seq) - 0.5
        log_stds_seq = jnp.clip(log_stds_seq, LOG_STD_MIN, LOG_STD_MAX)

        # Build mixture distribution
        mix_dist = tfd.Categorical(logits=logits_seq)
        comp_dist = tfd.MultivariateNormalDiag(
            loc=means_seq,
            scale_diag=jnp.exp(log_stds_seq)
        )
        dist = tfd.MixtureSameFamily(
            mixture_distribution=mix_dist,
            components_distribution=comp_dist
        )
        return dist

    @partial(jax.jit, static_argnums=(0))
    def eval_forward(self, x):
        return self.__call__(x)




