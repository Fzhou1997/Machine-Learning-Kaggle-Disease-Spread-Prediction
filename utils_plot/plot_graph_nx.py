import networkx as nx
import matplotlib.pyplot as plt


def plot_graph_nx(graph: nx.Graph,
                  node_attribute: str,
                  color_dict: dict,
                  figsize: tuple[float, float] = (12, 8),
                  title: str = None,
                  random_state: int = 42):
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(graph, seed=random_state)
    node_values = [graph.nodes[node][node_attribute] for node in graph.nodes]
    node_colors = [color_dict[node_value] for node_value in node_values]
    nx.draw(graph, pos=pos, with_labels=False, node_size=10, edge_color='gray', node_color=node_colors)
    if title is not None:
        plt.title(title)
    plt.show()