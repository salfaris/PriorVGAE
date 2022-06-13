import jax
import jax.numpy as jnp
import jraph
import networkx as nx
import numpy as onp

import pickle
from typing import Tuple

def load_dataset(path_to_dataset: str) -> jraph.GraphsTuple:
    with open(path_to_dataset, 'rb') as f:
        cora_ds = pickle.load(f)
    return cora_ds


def convert_jraph_to_networkx_graph(jraph_graph: jraph.GraphsTuple) -> nx.Graph:
  """Converts a JAX GraphsTuple to a NetworkX graph.
  
  Based fully on: 
  https://github.com/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb
  """
  nodes, edges, receivers, senders, _, _, _ = jraph_graph
  nx_graph = nx.DiGraph()
  if nodes is None:
    for n in range(jraph_graph.n_node[0]):
      nx_graph.add_node(n)
  else:
    for n in range(jraph_graph.n_node[0]):
      nx_graph.add_node(n, node_feature=nodes[n])
  if edges is None:
    for e in range(jraph_graph.n_edge[0]):
      nx_graph.add_edge(int(senders[e]), int(receivers[e]))
  else:
    for e in range(jraph_graph.n_edge[0]):
      nx_graph.add_edge(
          int(senders[e]), int(receivers[e]), edge_feature=edges[e])
  return nx_graph


def draw_jraph_graph_structure(jraph_graph: jraph.GraphsTuple) -> None:
  nx_graph = convert_jraph_to_networkx_graph(jraph_graph)
  pos = nx.spring_layout(nx_graph)
  nx.draw(
      nx_graph, pos=pos, with_labels=True, node_size=500, font_color='yellow')


def compute_norm_and_weights(graph: jraph.GraphsTuple) -> Tuple[float, float]:
  graph_n_node = graph.n_node.item()
  graph_adj = nx.to_numpy_matrix(convert_jraph_to_networkx_graph(graph))
  adj_sum = onp.sum(graph_adj)
  pos_weight = float(graph_n_node**2 - adj_sum) / adj_sum
  norm_adj = graph_n_node**2 / 2.0*(graph_n_node**2 - adj_sum)
  return pos_weight, norm_adj