import networkx as nx
import matplotlib.pyplot as plt

def plot_graph_nx(graph: nx.Graph,
                  node_attribute: str,
                  color_dict: dict,
                  figsize: tuple[float, float] = (12, 8),
                  title: str = None,
                  random_state: int = 42):
    """
    Plot a graph using NetworkX and Matplotlib.

    Args:
        graph (nx.Graph): The NetworkX graph to be plotted.
        node_attribute (str): The node attribute to determine node colors.
        color_dict (dict): A dictionary mapping node attribute values to colors.
        figsize (tuple[float, float], optional): Size of the figure. Default is (12, 8).
        title (str, optional): Title of the plot. Default is None.
        random_state (int, optional): Seed for the layout algorithm. Default is 42.

    Returns:
        None
    """
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(graph, seed=random_state)
    node_values = [graph.nodes[node][node_attribute] for node in graph.nodes]
    node_colors = [color_dict[node_value] for node_value in node_values]
    nx.draw(graph, pos=pos, with_labels=False, node_size=10, edge_color='gray', node_color=node_colors)
    if title is not None:
        plt.title(title)
    plt.show()