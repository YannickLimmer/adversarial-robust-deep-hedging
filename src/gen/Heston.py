from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.gen.Generator import EulerGenerator, DiffusionGeneratorConfig
from src.util.TimeUtil import TimeDiscretization, UniformTimeDiscretization
from src.util.torch_util.ParameterUtil import ExpParameter, TanhParameter, TransformedParameter


@dataclass
class HestonParameterSet:
    reversion_speed: float
    reversion_level: float
    vol_of_vol: float
    correlation: float
    drift: float = 0
    initial_vol: float = None

    @property
    def initial_vol_set(self):
        return self.initial_vol is not None

    @classmethod
    def from_json(cls, json: Dict):
        return cls(**json)


class HestonGeneratorConfig(DiffusionGeneratorConfig):

    def __init__(
            self,
            td: TimeDiscretization,
            default_pars: HestonParameterSet,
            include_vol_swap: bool = False,
    ):
        super().__init__(td, process_dim=(3 if include_vol_swap else 2), noise_dim=2)
        self.default_pars = default_pars
        self.include_vol_swap = include_vol_swap


class HestonGenerator(EulerGenerator[HestonGeneratorConfig]):

    def _register__(self):
        self.reversion_speed = ExpParameter(
            torch.tensor(self.config.default_pars.reversion_speed, dtype=self.dtype, device=self.device)
        )
        self.reversion_level = ExpParameter(
            torch.tensor(self.config.default_pars.reversion_level, dtype=self.dtype, device=self.device)
        )
        self.vol_of_vol = ExpParameter(
            torch.tensor(self.config.default_pars.vol_of_vol, dtype=self.dtype, device=self.device)
        )
        self.correlation = TanhParameter(
            torch.tensor(self.config.default_pars.correlation, dtype=self.dtype, device=self.device)
        )

        self.drift_par = TransformedParameter(
            torch.tensor(self.config.default_pars.drift, dtype=self.dtype, device=self.device),
            requires_grad=False,
        )
        if self.config.default_pars.initial_vol_set:
            self.initial_vol = TransformedParameter(
                torch.tensor(self.config.default_pars.initial_vol, dtype=self.dtype, device=self.device)
            )
        else:
            self.initial_vol = self.reversion_level

    def forward(
            self,
            batch_size: int,
            noise: Optional[torch.Tensor] = None,
            initial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        paths = super().forward(batch_size, noise, initial)
        return self.add_vol_swap(paths)

    def drift(self, t: float, prev: torch.Tensor) -> torch.Tensor:
        asset, var = torch.split(prev, 1, dim=1)
        asset_component = asset * self.drift_par.val
        volatility_component = self.reversion_speed.val * (self.reversion_level.val - var)
        return torch.cat([asset_component, volatility_component], dim=1)

    def diffusion(self, t: float, prev: torch.Tensor) -> torch.Tensor:
        asset, var = torch.split(prev, 1, dim=1)
        root_of_vol = torch.sqrt(torch.nn.functional.relu(var))
        asset_component = asset * root_of_vol
        volatility_component = self.vol_of_vol.val * root_of_vol
        return torch.cat([asset_component, volatility_component], dim=1)[:, :, None] * self.correlation_matrix_root

    @property
    def correlation_matrix_root(self) -> torch.Tensor:
        target_matrix = torch.eye(2, device=self.device, dtype=self.dtype)
        target_matrix[1, 1] = torch.sqrt(1 - self.correlation.val ** 2)
        target_matrix[1, 0] = self.correlation.val
        return target_matrix[None, :, :]

    @property
    def default_initial(self) -> torch.Tensor:
        return torch.cat([torch.ones(1, dtype=self.dtype, device=self.device), self.reversion_level.val[None]])

    def add_vol_swap(self, paths: torch.Tensor) -> torch.Tensor:
        if not self.config.include_vol_swap:
            return paths
        correction = self.calculate_correction(paths)
        times = torch.tensor(np.concatenate((np.array([0.0]), self.config.td.time_step_increments)), dtype=self.dtype, device=self.device)
        volatility_swap = torch.cumsum(paths[:, :, 1] * times, dim=1) + correction
        return torch.cat((paths, volatility_swap[:, :, None]), dim=2)

    def calculate_correction(self, paths: torch.Tensor) -> torch.Tensor:
        ttm = torch.tensor(self.config.td.time_to_maturity, dtype=self.dtype, device=self.device)
        return (paths[:, :, 1] - self.reversion_level.val) / self.reversion_speed.val \
            * (1 - torch.exp(- self.reversion_speed.val * ttm)) + self.reversion_level.val * ttm


if __name__ == '__main__':
    pars = HestonParameterSet(3.0, 0.03, 0.2, -.8)
    tdis = UniformTimeDiscretization.from_bounds(0, 1, 60)
    conf = HestonGeneratorConfig(tdis, pars, True)
    gen = HestonGenerator(conf)
    p = gen(100).detach()
    print(gen.default_initial)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 9))
    ax1.plot(tdis.times, p.numpy()[:, :, 0].T)
    ax2.plot(tdis.times, p.numpy()[:, :, 1].T)
    ax3.plot(tdis.times, p.numpy()[:, :, 2].T)
    plt.show()
