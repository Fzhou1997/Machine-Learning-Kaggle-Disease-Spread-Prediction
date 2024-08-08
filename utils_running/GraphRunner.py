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
        # Return class predictions (e.g., argmax for classification)
        return torch.argmax(out, dim=1)

    def predict_proba(self, graph_data: Data, device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> Tensor:
        self.model.to(device)
        self.model.eval()
        graph_data = graph_data.to(device)
        with torch.no_grad():
            out = self.model(graph_data.x, graph_data.edge_index)
        # Return probabilities for the positive class (assuming binary classification)
        return torch.softmax(out, dim=1)[:, 1]