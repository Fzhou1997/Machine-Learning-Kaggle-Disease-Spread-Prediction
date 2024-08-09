import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphConvolutionalNetwork(torch.nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels, dropout=0.5):
        super(GraphConvolutionalNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()

        for i, hidden_layer in enumerate(hidden_layers):
            if i == 0:
                self.layers.append(GCNConv(in_channels, hidden_layer))
            else:
                self.layers.append(GCNConv(hidden_layers[i - 1], hidden_layer))

        # Final output layer
        self.layers.append(GCNConv(hidden_layers[-1], out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.layers[-1](x, edge_index).squeeze()
        return torch.sigmoid(x)
