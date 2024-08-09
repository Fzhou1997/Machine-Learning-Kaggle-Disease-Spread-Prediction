from os import PathLike
from typing import Literal, Self

import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) implementation using PyTorch.

    Args:
        input_size (int): Number of input features.
        hidden_sizes (list[int]): List of hidden layer sizes.
        activations (list[Literal['relu', 'tanh', 'sigmoid']]): List of activation functions for each hidden layer.
        output_activation (Literal['sigmoid', 'softmax', 'none'], optional): Activation function for the output layer. Default is 'sigmoid'.
    """
    def __init__(self,
                 input_size: int,
                 hidden_sizes: list[int],
                 activations: list[Literal['relu', 'tanh', 'sigmoid']],
                 output_activation: Literal['sigmoid', 'softmax', 'none'] = 'sigmoid'):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size, activation in zip(hidden_sizes, activations):
            layers.append(nn.Linear(in_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f'Unknown activation function: {activation}')
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation != 'none':
            raise ValueError(f'Unknown output activation function: {output_activation}')
        self.model = nn.Sequential(*layers)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, input_size].

        Returns:
            torch.Tensor: Output tensor with shape [batch_size, 1].
        """
        return self.model(x)

    def predict(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Predict binary class labels for the input data.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, input_size].

        Returns:
            torch.Tensor: Predicted binary class labels with shape [batch_size, 1].
        """
        return torch.round(self.forward(x))

    def save(self,
             path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        """
        Save the model state to a file.

        Args:
            path (str | bytes | PathLike[str] | PathLike[bytes]): Path to the file where the model state will be saved.
        """
        torch.save(self.state_dict(), path)

    def load(self,
             path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        """
        Load the model state from a file.

        Args:
            path (str | bytes | PathLike[str] | PathLike[bytes]): Path to the file from which the model state will be loaded.

        Returns:
            Self: The model instance with the loaded state.
        """
        self.load_state_dict(torch.load(path))
        return self