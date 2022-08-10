import pickle
from typing import Tuple, Optional

import jax.numpy as jnp
import jraph
import networkx as nx
import numpy as np
from shapely.geometry import Polygon


def load_dataset(path_to_dataset: str) -> jraph.GraphsTuple:
    with open(path_to_dataset, 'rb') as f:
        cora_ds = pickle.load(f)
    return cora_ds


def generate_synthetic_dataset(
        x_dim: int = 15,
        y_dim: int = 10,
        scale: int = 1) -> Tuple[
            np.ndarray, jnp.ndarray, np.ndarray, int, Tuple[int, int]]:
    """Generates a synthetic dataset."""
    num_x = x_dim * scale
    num_y = y_dim * scale

    # Generate polygons.
    polygons = []
    for j in range(num_y):
        for i in range(num_x):
            coords = [(i, j), (i+1, j), (i+1, j+1), (i, j+1), (i, j)]
            polygons.append(Polygon(coords))

    num_regions = len(polygons)

    # Adjacency matrix.
    # NOTE: Do not make `A` a jnp.ndarray as for some reason it slows down
    #       the predictive model when used. Very strange.
    A = np.zeros(shape=(num_regions, num_regions))
    for i in range(num_regions):
        for j in range(i+1, num_regions):
            polygons_intersect = (
                polygons[i].intersection(polygons[j]).length > 0)
            if polygons_intersect:
                A[i, j] = A[j, i] = 1

    # Number of neighbours.
    d = A.sum(axis=0)
    D = jnp.diag(d)

    data_shape = (num_x, num_y)

    return A, D, d, num_regions, data_shape


def create_grid_graph(adj_matrix: np.ndarray) -> jraph.GraphsTuple:
    G = nx.from_numpy_matrix(adj_matrix)

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


def get_car_draws_as_graph(
        car_draws: jnp.ndarray,
        graph: Optional[jraph.GraphsTuple] = None,
        adj_matrix: Optional[jnp.ndarray] = None) -> jraph.GraphsTuple:
    if graph is None and adj_matrix is None:
        raise ValueError('Either graph or adj_matrix must be provided.')

    if graph is None:
        graph = create_grid_graph(adj_matrix=adj_matrix)

    return graph._replace(nodes=car_draws)


def get_car_draws_as_graph_given_base_graph(
        car_draws: jnp.ndarray,
        base_graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    return base_graph._replace(nodes=car_draws)


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
    adj_sum = np.sum(graph_adj)
    pos_weight = float(graph_n_node**2 - adj_sum) / adj_sum
    norm_adj = graph_n_node**2 / 2.0*(graph_n_node**2 - adj_sum)
    return pos_weight, norm_adj
