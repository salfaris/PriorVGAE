from typing import Tuple, NamedTuple, Callable

import haiku as hk
import jax
import jax.numpy as jnp
import jraph


class VGAEOutput(NamedTuple):
    mean: jnp.ndarray
    log_std: jnp.ndarray
    output: jraph.GraphsTuple


def encoder(
        graph: jraph.GraphsTuple,
        hidden_gnn_dim: int,
        hidden_fc_dim: int,
        latent_dim: int,
        batch_size: int = 32,
        num_nodes: int = 150,
        act_fn: Callable = jax.nn.relu) -> Tuple[jnp.ndarray, jnp.ndarray]:

    @jraph.concatenated_args
    def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
        """Node update function for hidden layer."""
        net = hk.Sequential([hk.Linear(hidden_gnn_dim), act_fn])
        return net(feats)

    # Graph layer
    net_hidden = jraph.GraphConvolution(
        update_node_fn=node_update_fn,
        add_self_edges=True
    )
    graph = net_hidden(graph)

    # Fully-connected layer
    x = graph.nodes.reshape(
        batch_size, num_nodes, hidden_gnn_dim)
    x = hk.Flatten()(x)
    x = hk.Linear(hidden_fc_dim)(x)
    x = jax.nn.relu(x)

    mean = hk.Linear(latent_dim, name='mean')(x)
    log_std = hk.Linear(latent_dim, name='log_std')(x)
    return mean, log_std


def decoder(
        z: jnp.ndarray,
        graph: jraph.GraphsTuple,
        hidden_fc_dim: int,
        hidden_gnn_dim: int,
        output_dim: int,
        batch_size: int = 32,
        num_nodes: int = 150,
        act_fn: Callable = jax.nn.relu) -> jraph.GraphsTuple:
    # `hidden_gnn_dim` must match node dim
    # from node_update_fn of last GNN layer.
    z = hk.Linear(hidden_fc_dim,
                  name='decoder_hidden1_fc')(z)
    z = hk.Linear(num_nodes*hidden_gnn_dim,
                  name='decoder_hidden2_fc')(z)  # (batch_size, num_nodes*hidden_gnn_dim)
    z = act_fn(z)
    # Reshape to jraph.batch format: (batch_size*num_nodes, hidden_gnn_dim)
    z = z.reshape((batch_size*num_nodes, hidden_gnn_dim))
    graph = graph._replace(nodes=z)

    @jraph.concatenated_args
    def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
        """Node update function for hidden layer."""
        net = hk.Sequential(
            [hk.Linear(hidden_gnn_dim,
                       name='decoder_hidden_gnn'), act_fn])
        return net(feats)

    net = jraph.GraphConvolution(
        update_node_fn=node_update_fn,
        add_self_edges=True
    )
    graph = net(graph)
    net = jraph.GraphConvolution(
        update_node_fn=hk.Linear(output_dim, name='decoder_output')
    )
    graph = net(graph)
    return graph


class VGAE(hk.Module):
    """VGAE network definition."""

    def __init__(
        self,
        hidden_gnn_dim: int,
        hidden_fc_dim: int,
        latent_dim: int,
        output_dim: int,
        batch_size: int,
        num_nodes: int = 150,
    ):
        super().__init__()
        self._hidden_gnn_dim = hidden_gnn_dim
        self._hidden_fc_dim = hidden_fc_dim
        self._latent_dim = latent_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._num_nodes = num_nodes
        self.act = jax.nn.relu

    def __call__(self, graph: jraph.GraphsTuple) -> VGAEOutput:
        mean, log_std = encoder(
            graph,
            self._hidden_gnn_dim,
            self._hidden_fc_dim,
            self._latent_dim,
            self._batch_size,
            self._num_nodes,
            self.act,
        )

        std = jnp.exp(log_std)
        z = mean + std * jax.random.normal(hk.next_rng_key(), mean.shape)

        output = decoder(
            z,
            graph,
            self._hidden_fc_dim,
            self._hidden_gnn_dim,
            self._output_dim,
            self._batch_size,
            self._num_nodes,
            self.act
        )

        return VGAEOutput(mean, log_std, output)
