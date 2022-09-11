from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass
class StrategyNetConfig:

    """
    Configures a strategy layer.

    :param dimension_of_asset: The dimension of the asset.
    :type dimension_of_asset: int
    :param number_of_layers: The number of intermediate layers.
    :type number_of_layers: int
    :param nodes_in_intermediate_layers: The number of nodes used in intermediate layers.
    :type nodes_in_intermediate_layers: int
    :param intermediate_activation: The activation used for intermediate layers. Defaults to nn.ReLU().
    :type intermediate_activation: nn.Module
    :param output_activation: The activation used for the output. Defaults to nn.Identity().
    :type output_activation: nn.Module
    """

    dimension_of_asset: int
    number_of_layers: int
    nodes_in_intermediate_layers: int
    intermediate_activation: nn.Module = nn.ReLU()
    output_activation: nn.Module = nn.Identity()


class StrategyNet(nn.Module):

    """
    The neural net that is trained in a deep hedge and yields the strategy.

    The network is constructed with the components provided by the config and consists of three elements:
        - `input_layer`: A linear transformation from input dimension to the intermediate layer dimension. The input
        dimension depends on the hedging problem and is one larger than the dimension of the asset.
        - `intermediate_layers`: A collection of linear layers that maps from intermediate layer dimension to
        intermediate layer dimension.
        - `output_layer`: A linear transformation form the intermediate layer dimension to the output dimension.

    :param config: The configuration of the strategy net.
    :type config: StrategyNetConfig
    """

    def __init__(self, config: StrategyNetConfig):
        super().__init__()
        self.config = config

        self.input_layer = nn.Linear(self.config.dimension_of_asset + 1, self.config.nodes_in_intermediate_layers)

        self.intermediate_layers = nn.ModuleDict({
            f'IntermediateLayer{layer_number}': nn.Linear(
                self.config.nodes_in_intermediate_layers,
                self.config.nodes_in_intermediate_layers,
            ) for layer_number in range(self.config.number_of_layers)
        })

        self.output_layer = nn.Linear(self.config.nodes_in_intermediate_layers, self.config.dimension_of_asset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward call of the strategy layer. Applies the input layer, then iterates through the intermediate layers
        which are applied posterior to the configured activation. Eventually, the output activation and output layer are
        applied.

        :param x: Input tensor of shape (n, d+1,), where n is the batch size and d the asset dimension.
        :type x: torch.Tensor
        :return: Output tensor, the strategy, of shape (n, d,), where n is the batch size and d the asset dimension.
        :rtype: torch.Tensor
        """
        x = self.input_layer(x)
        for intermediate_layer in self.intermediate_layers.values():
            x = self.config.intermediate_activation(x)
            x = intermediate_layer(x)
        x = self.config.output_activation(x)
        return self.output_layer(x)
