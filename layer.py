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

# GAT implementation adapted from https://github.com/deepmind/jraph/blob/master/jraph/_src/models.py#L442.


def GAT(attention_query_fn: Callable,
        attention_logit_fn: Callable,
        node_update_fn: Optional[Callable] = None,
        add_self_edges: bool = False) -> Callable:
    """Returns a method that applies a Graph Attention Network layer.

    Graph Attention message passing as described in
    https://arxiv.org/pdf/1710.10903.pdf. This model expects node features as a
    jnp.array, may use edge features for computing attention weights, and
    ignore global features. It does not support nests.
    Args:
      attention_query_fn: function that generates attention queries from sender
        node features.
      attention_logit_fn: function that converts attention queries into logits for
        softmax attention.
      node_update_fn: function that updates the aggregated messages. If None, will
        apply leaky relu and concatenate (if using multi-head attention).

    Returns:
      A function that applies a Graph Attention layer.
    """
    # pylint: disable=g-long-lambda
    if node_update_fn is None:
        # By default, apply the leaky relu and then concatenate the heads on the
        # feature axis.
        def node_update_fn(x): return jnp.reshape(
            jax.nn.leaky_relu(x), (x.shape[0], -1))

    def _ApplyGAT(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Applies a Graph Attention layer."""
        nodes, edges, receivers, senders, _, _, _ = graph
        # Equivalent to the sum of n_node, but statically known.
        try:
            sum_n_node = nodes.shape[0]
        except IndexError:
            raise IndexError('GAT requires node features')

        # Pass nodes through the attention query function to transform
        # node features, e.g. with an MLP.
        nodes = attention_query_fn(nodes)

        total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]
        if add_self_edges:
            # We add self edges to the senders and receivers so that each node
            # includes itself in aggregation.
            receivers, senders = add_self_edges_fn(receivers, senders,
                                                   total_num_nodes)

        # We compute the softmax logits using a function that takes the
        # embedded sender and receiver attributes.
        sent_attributes = nodes[senders]
        received_attributes = nodes[receivers]
        att_softmax_logits = attention_logit_fn(sent_attributes,
                                                received_attributes, edges)

        # Compute the attention softmax weights on the entire tree.
        att_weights = jraph.segment_softmax(
            att_softmax_logits, segment_ids=receivers, num_segments=sum_n_node)

        # Apply attention weights.
        messages = sent_attributes * att_weights
        # Aggregate messages to nodes.
        nodes = jax.ops.segment_sum(
            messages, receivers, num_segments=sum_n_node)

        # Apply an update function to the aggregated messages.
        nodes = node_update_fn(nodes)

        return graph._replace(nodes=nodes)

    # pylint: enable=g-long-lambda
    return _ApplyGAT


# Adapted from jraph https://github.com/deepmind/jraph/blob/31cf4117e68e8cd26a3c2628ea03a2074ff6157c/jraph/_src/models.py#L514
def GraphConvolutionSharpening(
        update_node_fn: Callable[[ArrayTree], ArrayTree],
        aggregate_nodes_fn: AggregateEdgesToNodesFn = jax.ops.segment_sum,
        add_self_edges: bool = False,
        symmetric_normalization: bool = True):
    """Returns a method that applies a Graph Convolution layer.
    Graph Convolutional layer as in https://arxiv.org/abs/1609.02907,
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
      A method that applies a Graph Convolution layer.
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
