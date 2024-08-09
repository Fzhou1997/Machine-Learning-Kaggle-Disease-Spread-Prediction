import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GraphAttentionNetwork(torch.nn.Module):
    """
    A Graph Attention Network (GAT) implementation using PyTorch and PyTorch Geometric.

    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden units in the GAT layer.
        out_channels (int): Number of output features per node.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(GraphAttentionNetwork, self).__init__()
        # First GAT layer with multiple attention heads
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=0.6)
        # Second GAT layer with a single attention head
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        """
        Forward pass of the GAT model.

        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity matrix with shape [2, num_edges].

        Returns:
            torch.Tensor: Output node features with shape [num_nodes, out_channels].
        """
        # Apply the first GAT layer and ELU activation
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        # Apply the second GAT layer
        x = self.conv2(x, edge_index)
        # Apply sigmoid activation and squeeze the last dimension
        return torch.sigmoid(x).squeeze(-1)