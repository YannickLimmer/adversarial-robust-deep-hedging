from dataclasses import dataclass, field

import torch
from torch import nn

from src.generator.Coefficient import CoefficientConfig, Coefficient
from src.util.processes.HestonGenerator import HestonParameterSet


@dataclass
class HestonCoefficientConfig(CoefficientConfig):
    initializer: HestonParameterSet
    initial_asset_price: float

    time_invariant: bool = field(default=True, init=False)


class HestonDriftCoefficient(Coefficient[HestonCoefficientConfig]):

    def __init__(self, config: HestonCoefficientConfig):
        super().__init__(config)

        self.drift = nn.Parameter(torch.tensor(self.config.initializer.drift, dtype=torch.float32))
        self.reversion_speed = nn.Parameter(torch.tensor(self.config.initializer.reversion_speed, dtype=torch.float32))
        self.reversion_level = nn.Parameter(torch.tensor(self.config.initializer.reversion_level, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.eliminate_time_if_time_invariant(x)
        asset_component = x[:, 0] * self.drift
        volatility_component = self.reversion_speed * (self.reversion_level - x[:, 1])
        return torch.stack([asset_component, volatility_component], dim=1)

    def get_initial_asset_price(self) -> torch.Tensor:
        target_tensor = torch.ones(2) * self.config.initial_asset_price
        target_tensor[1] = self.reversion_level
        return target_tensor


class HestonDiffusionCoefficient(Coefficient[HestonCoefficientConfig]):

    def __init__(self, config: HestonCoefficientConfig):
        super().__init__(config)

        self.vol_of_vol = nn.Parameter(torch.tensor(self.config.initializer.vol_of_vol, dtype=torch.float32))
        self.correlation = nn.Parameter(torch.tensor(self.config.initializer.correlation, dtype=torch.float32))

    @property
    def correlation_matrix_root(self) -> torch.Tensor:
        target_matrix = torch.eye(2)
        target_matrix[1, 1] = torch.sqrt(1 - self.correlation ** 2)
        target_matrix[0, 1] = self.correlation
        return target_matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.eliminate_time_if_time_invariant(x)
        root_of_vola = torch.sqrt(torch.abs(x[:, 1]))
        asset_component = x[:, 0] * root_of_vola
        volatility_component = self.vol_of_vol * root_of_vola
        return torch.stack([asset_component, volatility_component], dim=1)[:, :, None] \
            * self.correlation_matrix_root[None, :, :]

