from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from src.config import DEVICE
from src.generator.HestonCoefficient import EPS
from src.generator.SdeGenerator import SdeGenerator, GeneratorConfig


@dataclass
class rBergomiGeneratorConfig(GeneratorConfig):
    pass


class rBergomiGenerator(SdeGenerator[rBergomiGeneratorConfig]):

    def __init__(self, generator_config: rBergomiGeneratorConfig):
        super().__init__(generator_config)

        self._correlation = nn.Parameter(
            torch.tensor(self.config.correlation, dtype=torch.float32, device=DEVICE),
        )
        self.alpha = nn.Parameter(
            torch.tensor(self.config.alpha, dtype=torch.float32, device=DEVICE),
        )

    @property
    def initial_asset_price(self) -> Callable[[], torch.Tensor]:
        return lambda x: x

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        brownian_increments = self.generate_brownian_increments(noise)

    def generate_brownian_increments(self, noise: torch.Tensor) -> torch.Tensor:
        return noise @ self.covariance_matrix

    @property
    def correlation(self):
        return torch.clamp(
            self._correlation,
            torch.tensor([-1 + EPS], device=DEVICE),
            torch.tensor([1 - EPS], device=DEVICE),
        )

    @property
    def covariance_matrix(self) -> torch.Tensor:
        target_matrix = torch.eye(2, device=DEVICE)
        target_matrix[0, 0] = self.config.td.time_step_increments[0]
        target_matrix[0, 1] = target_matrix[1, 0] = 1.0 / (1.0 * self.alpha + 1) * \
            self.config.td.time_step_increments[0] ** (1.0 + self.alpha)
        target_matrix[1, 1] = 1.0 / (2.0 * self.alpha + 1) * \
            self.config.td.time_step_increments[0] ** (1.0 + 2 * self.alpha)
        return target_matrix
