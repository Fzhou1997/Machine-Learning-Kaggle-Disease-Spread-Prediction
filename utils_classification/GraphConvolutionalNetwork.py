import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphConvolutionalNetwork(torch.nn.Module):
    """
    A Graph Convolutional Network (GCN) implementation using PyTorch and PyTorch Geometric.

    Args:
        in_channels (int): Number of input features per node.
        hidden_layers (list of int): List containing the number of hidden units for each hidden layer.
        out_channels (int): Number of output features per node.
        dropout (float, optional): Dropout rate. Default is 0.5.
    """
    def __init__(self, in_channels, hidden_layers, out_channels, dropout=0.5):
        super(GraphConvolutionalNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()

        # Create GCN layers
        for i, hidden_layer in enumerate(hidden_layers):
            if i == 0:
                self.layers.append(GCNConv(in_channels, hidden_layer))
            else:
                self.layers.append(GCNConv(hidden_layers[i - 1], hidden_layer))

        # Final output layer
        self.layers.append(GCNConv(hidden_layers[-1], out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN model.

        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity matrix with shape [2, num_edges].

        Returns:
            torch.Tensor: Output node features with shape [num_nodes, out_channels].
        """
        # Apply GCN layers with ReLU activation and dropout
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply the final GCN layer and sigmoid activation
        x = self.layers[-1](x, edge_index).squeeze()
        return torch.sigmoid(x)