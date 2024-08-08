import torch.nn.functional as F
from torch_geometric.nn import GCN, GCNConv
import torch

class GraphConvolutionalNetwork(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphConvolutionalNetwork, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index).squeeze()
        return x