import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(graph: nx.Graph,
               title: str = None,
               random_state: int = 42):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=random_state)
    nx.draw(graph, pos=pos, with_labels=False, node_size=10, edge_color='gray')
    if title is not None:
        plt.title(title)
    plt.show()