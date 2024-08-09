from typing import Literal
import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data, DataLoader

class GraphRunner:
    def __init__(self, model: Module):
        self.model = model

    def predict(self, graph: Data, device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> Tensor:
        self.model.to(device)
        self.model.eval()
        graph = graph.to(device)
        with torch.no_grad():
            out = self.model(graph.x.to(device), graph.edge_index.to(device))
        return (out > 0.5).float()

    def predict_proba(self, graph: Data, device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> Tensor:
        self.model.to(device)
        self.model.eval()
        graph = graph.to(device)
        with torch.no_grad():
            out = self.model(graph.x.to(device), graph.edge_index.to(device))
        return torch.sigmoid(out)