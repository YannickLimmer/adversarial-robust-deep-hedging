from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass
class StrategyNetConfig:
    dimension_of_asset: int
    number_of_layers: int
    nodes_in_intermediate_layers: int
    intermediate_activation: nn.Module = nn.ReLU()
    output_activation: nn.Module = nn.Identity()


class StrategyNet(nn.Module):

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
        x = self.input_layer(x)
        for intermediate_layer in self.intermediate_layers.values():
            x = self.config.intermediate_activation(x)
            x = intermediate_layer(x)
        x = self.config.output_activation(x)
        return self.output_layer(x)
