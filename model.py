import haiku as hk
import jax
import jax.numpy as jnp
import jraph

from typing import Tuple, NamedTuple

class VGAEOutput(NamedTuple):
  mean: jnp.ndarray
  log_std: jnp.ndarray
  output: jraph.GraphsTuple
  
class VGAE(hk.Module):
  def __init__(
    self,
    hidden_dim: int,
    latent_dim: int,
    output_dim: int,
  ):
    super().__init__()
    self._hidden_dim = hidden_dim
    self._latent_dim = latent_dim
    self._output_dim = output_dim
  
  def __call__(self, graph: jraph.GraphsTuple) -> VGAEOutput:
    mean_graph, log_std_graph = vgae_encoder(
      graph, self._hidden_dim, self._latent_dim
    )
    mean, log_std = mean_graph.nodes, log_std_graph.nodes
    std = jnp.exp(log_std)
    z = mean + std * jax.random.normal(hk.next_rng_key(), mean.shape)
    z_graph = mean_graph._replace(nodes=z)  # this step assumes only node features are used.
    output = prior_decode(z_graph, self._hidden_dim, self._output_dim)

    return VGAEOutput(mean, log_std, output)

def vgae_encoder(graph: jraph.GraphsTuple,
                 hidden_dim: int,
                 latent_dim: int) -> Tuple[jraph.GraphsTuple, jraph.GraphsTuple]:
  """VGAE network definition."""
  graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))
  
  @jraph.concatenated_args
  def hidden_node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Node update function for hidden layer."""
    net = hk.Sequential([hk.Linear(hidden_dim), jax.nn.elu])
    return net(feats)

  @jraph.concatenated_args
  def latent_node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Node update function for latent layer."""
    return hk.Linear(latent_dim)(feats)

  net_hidden = jraph.GraphConvolution(
    update_node_fn=hidden_node_update_fn,
    add_self_edges=True
  )
  h_graph = net_hidden(graph)
  net_hidden = jraph.GraphConvolution(
    update_node_fn=hidden_node_update_fn,
    add_self_edges=True
  )
  h_graph = net_hidden(graph)
  
  net_mean = jraph.GraphConvolution(
    update_node_fn=latent_node_update_fn
  )
  net_log_std = jraph.GraphConvolution(
    update_node_fn=latent_node_update_fn
  )
  mean_graph, log_std_graph = net_mean(h_graph), net_log_std(h_graph)
  return mean_graph, log_std_graph


def prior_decode(graph: jraph.GraphsTuple,
                 hidden_dim: jnp.ndarray,
                 output_dim: jnp.ndarray) -> jnp.ndarray:
  
  # @jraph.concatenated_args
  # def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    # net = hk.Sequential([hk.Linear(hidden_dim, name='decoder_hidden'), 
    #                      jax.nn.elu, 
    #                      hk.Linear(output_dim, name='decoder_output')])
  #   return net(feats)
  
  net = jraph.GraphConvolution(
    update_node_fn=lambda x: jax.nn.elu(
      hk.Linear(hidden_dim, name='decoder_hidden')(x)),
    add_self_edges=True)
  graph = net(graph)
  net = jraph.GraphConvolution(
    update_node_fn=hk.Linear(output_dim, name='decoder_output'))
  return net(graph)


def inner_product_decode(pred_graph_nodes: jnp.ndarray, senders: jnp.ndarray,
           receivers: jnp.ndarray) -> jnp.ndarray:
  """Given a set of candidate edges, take dot product of respective nodes.

  Args:
    pred_graph_nodes: input graph nodes Z.
    senders: Senders of candidate edges.
    receivers: Receivers of candidate edges.

  Returns:
    For each edge, computes dot product of the features of the two nodes.

  """
  return jnp.squeeze(
      jnp.sum(pred_graph_nodes[senders] * pred_graph_nodes[receivers], axis=1))