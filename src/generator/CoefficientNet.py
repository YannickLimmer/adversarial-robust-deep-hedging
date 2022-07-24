from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from src.generator.Coefficient import CoefficientConfig, Coefficient


@dataclass
class CoefficientNetConfig(CoefficientConfig):
    number_of_layers: int
    nodes_in_intermediate_layers: int
    time_invariant: bool = False
    intermediate_activation: nn.Module = nn.ReLU()
    output_activation: nn.Module = nn.ReLU()
    transformation: Callable[[torch.Tensor], torch.Tensor] = nn.Identity()


class CoefficientNet(Coefficient[CoefficientNetConfig]):

    def __init__(self, config: CoefficientNetConfig):
        super(CoefficientNet, self).__init__(config)

        self.input_layer = nn.Linear(self.dimension_of_process, self.config.nodes_in_intermediate_layers)
        self.intermediate_layers = nn.ModuleDict(
            {
                f'IntermediateLayer{layer_number}': nn.Linear(
                    self.config.nodes_in_intermediate_layers,
                    self.config.nodes_in_intermediate_layers,
                ) for layer_number in range(self.config.number_of_layers)
            }
        )
        self.output_layer = nn.Linear(self.config.nodes_in_intermediate_layers, self.config.dimension_of_process)

    @property
    def dimension_of_process(self):
        return self.config.dimension_of_process + (0 if self.config.time_invariant else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(self.eliminate_time_if_time_invariant(x))
        for intermediate_layer in self.intermediate_layers.values():
            x = self.config.intermediate_activation(x)
            x = intermediate_layer(x)
        x = self.config.output_activation(x)
        return self.config.transformation(self.output_layer(x))


class DiffusionCoefficientNet(Coefficient[CoefficientNetConfig]):

    def __init__(self, config: CoefficientNetConfig):
        super().__init__(config)

        self.coefficients_by_noise_dimension = nn.ModuleDict(
            {
                f'NoiseDimension{noise_dim}': CoefficientNet(config)
                for noise_dim in range(self.config.dimension_of_process)
             }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for coefficient_for_noise_dimension in self.coefficients_by_noise_dimension.values():
            outputs.append(coefficient_for_noise_dimension(x))
        return torch.stack(outputs, dim=2)


