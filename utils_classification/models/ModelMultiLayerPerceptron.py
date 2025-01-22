import os
from os import PathLike
from typing import Self, Sequence, Type

import torch
import torch.nn as nn

from utils_torch.modules.LinearStack import LinearStack


class ModelMultiLayerPerceptron(nn.Module):
    _linear_stack: LinearStack

    def __init__(self,
                 num_layers: int,
                 num_features_in: int,
                 num_features_hidden: int | Sequence[int],
                 bias: bool | Sequence[bool] = True,
                 activation: Type[nn.Module] | Sequence[Type[nn.Module]] = nn.ReLU,
                 activation_kwargs: dict[str, float] | Sequence[dict[str, float]] = None,
                 dropout_p: float | Sequence[float] = 0.5,
                 dropout_inplace: bool | Sequence[bool] = False,
                 dropout_first: bool | Sequence[bool] = False,
                 batch_norm: bool | Sequence[bool] = False,
                 batch_norm_momentum: float | Sequence[float] = 0.1,
                 device: torch.device = None,
                 dtype: torch.dtype = torch.float32) -> None:
        super(ModelMultiLayerPerceptron, self).__init__()

        assert num_layers > 0, 'Number of layers must be greater than 0.'
        assert num_features_in > 0, "Number of input features must be greater than 0."
        if isinstance(num_features_hidden, int):
            num_features_hidden = [num_features_hidden] * (num_layers - 1)
        assert len(
            num_features_hidden) == num_layers - 1, "Number of hidden features must match the number of layers."
        num_features_out = 1
        num_features = [num_features_in] + num_features_hidden + [num_features_out]

        self._linear_stack = LinearStack(num_layers=num_layers,
                                         num_features=num_features,
                                         bias=bias,
                                         activation=activation,
                                         activation_kwargs=activation_kwargs,
                                         dropout_p=dropout_p,
                                         dropout_inplace=dropout_inplace,
                                         dropout_first=dropout_first,
                                         batch_norm=batch_norm,
                                         batch_norm_momentum=batch_norm_momentum,
                                         device=device,
                                         dtype=dtype)

    @property
    def num_layers(self) -> int:
        return self._linear_stack.num_layers

    @property
    def num_features_in(self) -> int:
        return self._linear_stack.linear_num_features_in

    @property
    def num_features_out(self) -> int:
        return 1

    @property
    def num_features_hidden(self) -> tuple[int, ...]:
        return self._linear_stack.linear_num_features[1:-1]

    @property
    def num_features(self) -> tuple[int, ...]:
        return self._linear_stack.linear_num_features

    @property
    def bias(self) -> tuple[bool, ...]:
        return self._linear_stack.linear_bias

    @property
    def activation(self) -> tuple[Type[nn.Module], ...]:
        return self._linear_stack.linear_activation

    @property
    def activation_kwargs(self) -> tuple[dict[str, float], ...]:
        return self._linear_stack.linear_activation_kwargs

    @property
    def dropout_p(self) -> tuple[float, ...]:
        return self._linear_stack.linear_dropout_p

    @property
    def dropout_inplace(self) -> tuple[bool, ...]:
        return self._linear_stack.linear_dropout_inplace

    @property
    def dropout_first(self) -> tuple[bool, ...]:
        return self._linear_stack.linear_dropout_first

    @property
    def batch_norm(self) -> tuple[bool, ...]:
        return self._linear_stack.linear_batch_norm

    @property
    def batch_norm_momentum(self) -> tuple[float, ...]:
        return self._linear_stack.linear_batch_norm_momentum

    @property
    def device(self) -> torch.device:
        return self._linear_stack.device

    @property
    def dtype(self) -> torch.dtype:
        return self._linear_stack.dtype

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return self._linear_stack.forward(x)

    def predict(self,
                logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def classify(self,
                 predicted: torch.Tensor,
                 threshold: float = 0.5) -> torch.Tensor:
        return (predicted > threshold).float()

    def save(self,
             model_dir: str | bytes | PathLike[str] | PathLike[bytes],
             model_name: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, f'{model_name}.pth'))

    def load(self,
             model_dir: str | bytes | PathLike[str] | PathLike[bytes],
             model_name: str) -> Self:
        self.load_state_dict(torch.load(os.path.join(model_dir, f'{model_name}.pth')))
        return self
