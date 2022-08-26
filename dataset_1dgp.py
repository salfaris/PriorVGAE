from typing import Callable, Optional, Type

import jax
import jax.numpy as jnp
import jraph
import networkx as nx
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

from tinygp.kernels import Matern32, ExpSquared


def euclidean_distance(x: jnp.ndarray, z: jnp.ndarray):
    """Euclidean distance.

    Original implementation: 
    https://github.com/elizavetasemenova/PriorVAE/blob/main/1d_gp.ipynb"""
    x, z = jnp.array(x), jnp.array(z)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(z.shape) == 1:
        z = z.reshape(-1, 1)
    assert x.shape[1] == z.shape[1]
    len_x, len_z = x.shape[0], z.shape[0]
    delta = jnp.zeros((len_x, len_z))
    for d in jnp.arange(x.shape[1]):
        x_d = x[:, d]
        z_d = z[:, d]
        delta += (x_d[:, jnp.newaxis] - z_d)**2
    return jnp.sqrt(delta)


def exp_sq_kernel(
        x: jnp.ndarray,
        z: jnp.ndarray,
        var: float,
        length: float,
        noise: float = 0.0,
        jitter: float = 1.0e-6) -> jnp.ndarray:
    """Squared Exponential Kernel a.k.a RBF Kernel.

    K(x, z) = var * exp(-0.5 * [(x - z) / length]^2)

    Modified from: 
    https://github.com/elizavetasemenova/PriorVAE/blob/main/1d_gp.ipynb
    """
    k = var * jnp.exp(-0.5 * (euclidean_distance(x, z) / length)**2.0)
    k += (noise+jitter) * jnp.eye(x.shape[0])
    return k


def matern_kernel(
        x: jnp.ndarray,
        z: jnp.ndarray,
        var: float,
        length: float,
        noise: float = 0.0,
        jitter: float = 1.0e-6) -> jnp.ndarray:
    """Squared Exponential Kernel a.k.a RBF Kernel.

    K(x, z) = var * exp(-0.5 * [(x - z) / length]^2)

    Modified from: 
    https://github.com/elizavetasemenova/PriorVAE/blob/main/1d_gp.ipynb
    """
    k = var * jnp.exp(-0.5 * (euclidean_distance(x, z) / length)**2.0)
    k += (noise+jitter) * jnp.eye(x.shape[0])
    return k


def gp_predictive_model(
        gp_kernel: Callable,
        x: jnp.ndarray,
        kernel_jitter: float = 1e-5,
        kernel_var: Optional[float] = None,
        kernel_length: Optional[float] = None,
        y: Optional[jnp.ndarray] = None,
        noise: bool = False):
    """Modified from: 
    https://github.com/elizavetasemenova/PriorVAE/blob/main/1d_gp.ipynb"""
    if kernel_length is None:
        kernel_length = numpyro.sample(
            'kernel_length', dist.InverseGamma(4, 1))
    if kernel_var is None:
        kernel_var = numpyro.sample('kernel_var', dist.LogNormal(0., 0.1))

    k = gp_kernel(x, x, kernel_var, kernel_length, kernel_jitter)

    if noise == False:
        numpyro.sample('y',  dist.MultivariateNormal(
            loc=jnp.zeros(x.shape[0]), covariance_matrix=k), obs=y)
    else:
        sigma = numpyro.sample('noise', dist.HalfNormal(0.1))
        f = numpyro.sample('f', dist.MultivariateNormal(
            loc=jnp.zeros(x.shape[0]), covariance_matrix=k))
        numpyro.sample('y', dist.Normal(f, sigma), obs=y)


def generate_gp_batch(
        rng: Type[jax.random.PRNGKey],
        x: jnp.ndarray,
        batch_size: int,
        kernel: Callable) -> jnp.ndarray:
    gp_predictive = Predictive(gp_predictive_model,
                               num_samples=batch_size)
    gp_draws = gp_predictive(
        rng,
        x=x,
        gp_kernel=kernel,
        kernel_jitter=1e-5)['y']
    return gp_draws


def gp_predictive_matern_model(
        x: jnp.ndarray,
        kernel_length: Optional[float] = None,
        y: Optional[jnp.ndarray] = None,
        noise: bool = False):
    if kernel_length is None:
        kernel_length = numpyro.sample(
            'kernel_length', dist.InverseGamma(4, 1))

    k = Matern32(scale=kernel_length)(x, x)

    if noise == False:
        numpyro.sample('y',  dist.MultivariateNormal(
            loc=jnp.zeros(x.shape[0]), covariance_matrix=k), obs=y)
    else:
        sigma = numpyro.sample('noise', dist.HalfNormal(0.1))
        f = numpyro.sample('f', dist.MultivariateNormal(
            loc=jnp.zeros(x.shape[0]), covariance_matrix=k))
        numpyro.sample('y', dist.Normal(f, sigma), obs=y)


def generate_gp_batch_matern(
        rng: Type[jax.random.PRNGKey],
        x: jnp.ndarray,
        batch_size: int) -> jnp.ndarray:
    gp_predictive = Predictive(gp_predictive_matern_model,
                               num_samples=batch_size)
    gp_draws = gp_predictive(
        rng,
        x=x)['y']
    return gp_draws


def create_1dgp_graph(num_locations: np.ndarray) -> jraph.GraphsTuple:
    G = nx.path_graph(num_locations)

    edges = list(G.edges)
    edges += [(edge[1], edge[0]) for edge in edges]
    senders = jnp.asarray([edge[0] for edge in edges])
    receivers = jnp.asarray([edge[1] for edge in edges])

    return jraph.GraphsTuple(
        n_node=jnp.asarray([len(G.nodes)]),
        n_edge=jnp.asarray([len(edges)]),
        nodes=None,
        edges=None,
        globals=None,
        senders=senders,
        receivers=receivers)
