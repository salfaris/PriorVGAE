from typing import Tuple, Callable, Optional

import jax
import jax.tree_util as tree
import jax.numpy as jnp
import jraph


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
