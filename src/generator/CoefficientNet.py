from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import nn

from src.generator.Coefficient import CoefficientConfig, Coefficient


@dataclass
class CoefficientNetConfig(CoefficientConfig):
    number_of_layers: int
    nodes_in_intermediate_layers: int
    dim_of_process: int
    intermediate_activation: nn.Module = None
    output_activation: nn.Module = None
    transformation: Callable[[torch.Tensor], torch.Tensor] = None

    def __post_init__(self):
        self.intermediate_activation = self.intermediate_activation if self.intermediate_activation else nn.ReLU()
        self.output_activation = self.output_activation if self.output_activation else nn.Identity()
        self.transformation = self.transformation if self.transformation else nn.Identity()

        self.dimension_of_process = self.dim_of_process


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


