from os import PathLike
from typing import Literal, Self

import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
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
        return self.model(x)

    def predict(self,
                x: torch.Tensor) -> torch.Tensor:
        return torch.round(self.forward(x))

    def save(self,
             path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        torch.save(self.state_dict(), path)

    def load(self,
             path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        self.load_state_dict(torch.load(path))
        return self
