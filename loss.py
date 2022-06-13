import jax.numpy as jnp

def compute_Lpq_loss(x: jnp.ndarray, y: jnp.ndarray, p: float, q: float) -> jnp.ndarray:
  """Computes the loss induced by the L_{p,q} norm.
  
  The L_{p,q} norm applies the Lp norm over features, and then
  the Lq norm over datapoints.
  """
  return jnp.power(
    jnp.sum(jnp.power(jnp.sum(jnp.power(x-y, p), axis=0), q/p)), 1/q)


def compute_L21_loss(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Computes the loss induced by the L_{2,1} norm.

  The L_{2,1} norm applies the L2 norm over features, and then 
  sum across datapoints.
  """
  return jnp.sum(jnp.sqrt(jnp.sum(jnp.square(x-y), axis=0)))


def compute_frobenius_loss(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Computes root squared error loss."""
  return jnp.sqrt(jnp.sum((jnp.square(x - y))))


def compute_mse_loss(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Computes mean squared error loss."""
  return jnp.mean(jnp.square(x - y))


def compute_kl_gaussian(mean: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
    r"""Calculate KL divergence between given and standard gaussian distributions.

    Args:
        mean: feature matrix of the mean.
        log_std: feature matrix of the log-covariance.

    Returns:
        A vector representing KL divergence of the two Gaussian distributions
        of length |V| where V is the nodes in the graph.
    """
    var = jnp.exp(log_std)
    return 0.5 * jnp.sum(
      -2*log_std - 1.0 + jnp.square(var) + jnp.square(mean), axis=-1)
  
  