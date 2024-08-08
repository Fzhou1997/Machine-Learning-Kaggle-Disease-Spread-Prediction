from typing import Literal
import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data, DataLoader

class GraphRunner:
    def __init__(self, model: Module):
        self.model = model

    def predict(self, graph_data: Data, device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> Tensor:
        self.model.to(device)
        self.model.eval()
        graph_data = graph_data.to(device)
        with torch.no_grad():
            out = self.model(graph_data.x, graph_data.edge_index)
        # Apply sigmoid to get probabilities and return class predictions
        probabilities = torch.sigmoid(out).squeeze()
        return (probabilities > 0.5).float()

    def predict_proba(self, graph_data: Data, device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> Tensor:
        self.model.to(device)
        self.model.eval()
        graph_data = graph_data.to(device)
        with torch.no_grad():
            out = self.model(graph_data.x, graph_data.edge_index)
        # Apply sigmoid to get probabilities
        return torch.sigmoid(out).squeeze()