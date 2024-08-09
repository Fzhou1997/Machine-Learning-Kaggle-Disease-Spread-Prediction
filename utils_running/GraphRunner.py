from typing import Literal
import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data, DataLoader

class GraphRunner:
    """
    A class to run predictions on graph data using a PyTorch model.

    Attributes:
        model (Module): The PyTorch model used for predictions.
    """

    def __init__(self, model: Module):
        """
        Initialize the GraphRunner with a PyTorch model.

        Args:
            model (Module): The PyTorch model to be used for predictions.
        """
        self.model = model

    def predict(self, graph: Data, device: Literal['cpu', 'cuda', 'mps'] = 'cpu', threshold: float = 0.5) -> Tensor:
        """
        Predict binary labels for the given graph data.

        Args:
            graph (Data): The graph data to be predicted.
            device (Literal['cpu', 'cuda', 'mps'], optional): The device to run the model on. Default is 'cpu'.
            threshold (float, optional): The threshold for converting probabilities to binary labels. Default is 0.5.

        Returns:
            Tensor: The predicted binary labels.
        """
        self.model.to(device)
        self.model.eval()
        graph = graph.to(device)
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index).squeeze()
        return (out > threshold).float()

    def predict_proba(self, graph: Data, device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> Tensor:
        """
        Predict probabilities for the given graph data.

        Args:
            graph (Data): The graph data to be predicted.
            device (Literal['cpu', 'cuda', 'mps'], optional): The device to run the model on. Default is 'cpu'.

        Returns:
            Tensor: The predicted probabilities.
        """
        self.model.to(device)
        self.model.eval()
        graph = graph.to(device)
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index).squeeze()
        return out