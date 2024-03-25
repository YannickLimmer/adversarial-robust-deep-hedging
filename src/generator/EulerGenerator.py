from dataclasses import dataclass
from typing import Callable, Generic

import torch
from torch import nn

from src.config import DEVICE
from src.generator.Coefficient import Coefficient, _Drift_Coefficient, _Diffusion_Coefficient
from src.generator.SdeGenerator import SdeGenerator, GeneratorConfig
from src.util.TimeUtil import TimeDiscretization


@dataclass
class EulerGeneratorConfig(GeneratorConfig):
    td: TimeDiscretization
    initial_asset_price: Callable[[], torch.Tensor]
    drift_coefficient: Coefficient
    diffusion_coefficient: Coefficient

    def __post_init__(self):
        if self.initial_asset_price is not None:
            for coefficient in (self.drift_coefficient, self.diffusion_coefficient):
                coefficient.config.dimension_of_process = self.initial_asset_price().shape[0]


class EulerGenerator(SdeGenerator[EulerGeneratorConfig], Generic[_Drift_Coefficient, _Diffusion_Coefficient]):

    def __init__(self, generator_config: EulerGeneratorConfig):
        super().__init__(generator_config)
        self.config = generator_config

        self.drift: _Drift_Coefficient = self.config.drift_coefficient
        self.diffusion: _Diffusion_Coefficient = self.config.diffusion_coefficient

        self.trainable_initial_asset_price = nn.Parameter(
            torch.ones(self.config.drift_coefficient.config.dimension_of_process),
        )

    @property
    def initial_asset_price(self) -> Callable[[], torch.Tensor]:

        def trainable_initial() -> torch.Tensor:
            return self.trainable_initial_asset_price

        return self.config.initial_asset_price if self.config.initial_asset_price is not None else trainable_initial

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generates a stochastic process with drift and diffusion components based on the given noise tensor.

        Args:
            noise (torch.Tensor): A tensor of shape (n, m, 1), where n is the number of paths, m is the number of time
            steps, and 1 is the number of dimensions of the Brownian motion. The tensor should contain random Gaussian
            noise used to generate the stochastic process.

        Returns:
            torch.Tensor: A tensor of shape (n, m+1), representing the generated stochastic process. Each row of
            the tensor corresponds to a different path of the process, and each column corresponds to a different time
            step. The first column of the tensor represents the initial value of the process at time 0.
        """
        process = [torch.ones_like(noise[:, 0, :], device=DEVICE) * self.initial_asset_price()]
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
        return torch.cat((time * torch.ones(arr.shape[0], 1, device=DEVICE), arr), dim=1)

