from dataclasses import dataclass
from typing import Union, Callable

import torch

from src.generator.Coefficient import Coefficient
from src.util.TimeUtil import TimeDiscretization


@dataclass
class GeneratorConfig:
    td: TimeDiscretization
    initial_asset_price: Callable[[], torch.Tensor]
    drift_coefficient: Coefficient
    diffusion_coefficient: Coefficient

    def __post_init__(self):
        for coefficient in (self.drift_coefficient, self.diffusion_coefficient):
            coefficient.config.dimension_of_process = self.initial_asset_price().shape[0]


class SdeGenerator(torch.nn.Module):

    def __init__(self, generator_config: GeneratorConfig):
        super().__init__()
        self.config = generator_config

        self.drift = self.config.drift_coefficient
        self.diffusion = self.config.diffusion_coefficient

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        process = [torch.ones_like(noise[:, 0, :]) * self.config.initial_asset_price()]
        for index in self.config.td.indices:
            process_coupled_with_time = self.couple_with_time(process[-1], self.config.td.times[index])
            drift_increment = self.drift(process_coupled_with_time) * self.config.td.time_step_increments[index]
            # Here, `diffusion` is of size (n, d, d,), `noise` of size (n, d, 1,), resulting in a multiplication of a
            # (d, d,) matrix with a (d,) column vector for every of the n entries.
            stoch_increment = self.diffusion(process_coupled_with_time) @ noise[:, index, :, None]
            process.append(process[-1] + drift_increment + stoch_increment[:, :, 0])
        return torch.stack(process, dim=1)

    @staticmethod
    def couple_with_time(arr: torch.Tensor, time: float) -> torch.Tensor:
        return torch.cat((time * torch.ones(arr.shape[0], 1), arr), dim=1)

