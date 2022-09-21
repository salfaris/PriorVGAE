from typing import Tuple, Callable, Optional, Union, Iterable, Mapping, Any

import jax
import jax.tree_util as tree
import jax.numpy as jnp
import jraph

# As of 04/2020 pytype doesn't support recursive types.
# pytype: disable=not-supported-yet
ArrayTree = Union[jnp.ndarray,
                  Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]

# All features will be an ArrayTree.
NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = Globals = ArrayTree

# Signature:
# (edges of each node to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToNodesFn = Callable[
    [EdgeFeatures, jnp.ndarray, int], NodeFeatures]


def add_self_edges_fn(receivers: jnp.ndarray, senders: jnp.ndarray,
                      total_num_nodes: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Adds self edges. Assumes self edges are not in the graph yet."""
    receivers = jnp.concatenate(
        (receivers, jnp.arange(total_num_nodes)), axis=0)
    senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)), axis=0)
    return receivers, senders


def GraphConvolutionSharpening(
        update_node_fn: Callable[[ArrayTree], ArrayTree],
        aggregate_nodes_fn: AggregateEdgesToNodesFn = jax.ops.segment_sum,
        add_self_edges: bool = False,
        symmetric_normalization: bool = True):
    """Returns a method that applies a Laplacian sharpening Graph Convolution layer.
    Implementation adapted from jraph:
        https://github.com/deepmind/jraph/blob/31cf4117e68e8cd26a3c2628ea03a2074ff6157c/jraph/_src/models.py#L514    
    NOTE: This implementation does not add an activation after aggregation.
    If you are stacking layers, you may want to add an activation between
    each layer.
    Args:
      update_node_fn: function used to update the nodes. In the paper a single
        layer MLP is used.
      aggregate_nodes_fn: function used to aggregates the sender nodes.
      add_self_edges: whether to add self edges to nodes in the graph as in the
        paper definition of GCN. Defaults to False.
      symmetric_normalization: whether to use symmetric normalization. Defaults
        to True. Note that to replicate the fomula of the linked paper, the
        adjacency matrix must be symmetric. If the adjacency matrix is not
        symmetric the data is prenormalised by the sender degree matrix and post
        normalised by the receiver degree matrix.
    Returns:
      A method that applies a Laplacian sharpening Graph Convolution layer.
    """
    def _ApplyGCN(graph):
        """Applies a Graph Convolution layer."""
        nodes, _, receivers, senders, _, _, _ = graph

        # First pass nodes through the node updater.
        nodes = update_node_fn(nodes)
        # Equivalent to jnp.sum(n_node), but jittable
        total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]
        if add_self_edges:
            # We add self edges to the senders and receivers so that each node
            # includes itself in aggregation.
            # In principle, a `GraphsTuple` should partition by n_edge, but in
            # this case it is not required since a GCN is agnostic to whether
            # the `GraphsTuple` is a batch of graphs or a single large graph.
            conv_receivers, conv_senders = add_self_edges_fn(receivers,
                                                             senders,
                                                             total_num_nodes)
            conv_receivers, conv_senders = add_self_edges_fn(conv_receivers,
                                                             conv_senders,
                                                             total_num_nodes)

        else:
            conv_senders = senders
            conv_receivers = receivers

        # pylint: disable=g-long-lambda
        if symmetric_normalization:
            # Calculate the normalization values.
            def count_edges(x): return jax.ops.segment_sum(
                jnp.ones_like(conv_senders), x, total_num_nodes)
            sender_degree = count_edges(conv_senders)
            receiver_degree = count_edges(conv_receivers)

            # Pre normalize by sqrt sender degree.
            # Avoid dividing by 0 by taking maximum of (degree, 1).
            nodes = tree.tree_map(
                lambda x: x *
                jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, None],
                nodes,
            )
            # Aggregate the pre normalized nodes.
            nodes = tree.tree_map(
                lambda x: aggregate_nodes_fn(x[conv_senders], conv_receivers,
                                             total_num_nodes), nodes)
            nodes = - nodes
            # Post normalize by sqrt receiver degree.
            # Avoid dividing by 0 by taking maximum of (degree, 1).
            nodes = tree.tree_map(
                lambda x:
                (x * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))
                 [:, None]),
                nodes,
            )
        else:
            nodes = tree.tree_map(
                lambda x: aggregate_nodes_fn(x[conv_senders], conv_receivers,
                                             total_num_nodes), nodes)
        # pylint: enable=g-long-lambda
        return graph._replace(nodes=nodes)

    return _ApplyGCN


def GraphwiseLayerNorm():
    """Applies graph-wise layer normalization to a graph layer."""

    def _ApplyLayerNorm(graph: jraph.GraphsTuple):
        # Sums over the nodes, i.e., axis=0.
        layer_mean = jnp.mean(graph.nodes, axis=0)
        layer_std = jnp.sqrt(jnp.mean(jnp.square(graph.nodes - layer_mean), axis=0)
                             )
        nodes = (graph.nodes - layer_mean) / layer_std
        return graph._replace(nodes=nodes)

    return _ApplyLayerNorm


def NodewiseLayerNorm():
    """Applies node-wise layer normalization to a graph layer.

    WARNING: Cannot be applied to a single feature hidden layer!
    """

    def _ApplyLayerNorm(graph: jraph.GraphsTuple):
        # Sums over the nodes, i.e., axis=0.
        layer_mean = jnp.mean(graph.nodes, axis=1).reshape(-1, 1)
        layer_std = jnp.sqrt(
            jnp.mean(jnp.square(graph.nodes - layer_mean), axis=1)).reshape(-1, 1)
        nodes = (graph.nodes - layer_mean) / layer_std
        return graph._replace(nodes=nodes)

    return _ApplyLayerNorm
