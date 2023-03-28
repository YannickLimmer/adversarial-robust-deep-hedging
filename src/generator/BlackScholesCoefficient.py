from dataclasses import dataclass, field

import torch
from torch import nn

from src.config import DEVICE
from src.generator.Coefficient import CoefficientConfig, Coefficient
from src.util.processes.BlackScholesGenerator import BlackScholesParameterSet
from src.util.processes.HestonGenerator import HestonParameterSet


@dataclass
class BlackScholesCoefficientConfig(CoefficientConfig):
    initializer: BlackScholesParameterSet
    initial_asset_price: float

    time_invariant: bool = field(default=True, init=False)


class BlackScholesDriftCoefficient(Coefficient[BlackScholesCoefficientConfig]):

    def __init__(self, config: BlackScholesCoefficientConfig):
        super().__init__(config)
        self.drift = nn.Parameter(torch.tensor(
                self.config.initializer.drift,
                dtype=torch.float32,
                device=DEVICE,
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.eliminate_time_if_time_invariant(x)
        return x * self.drift

    def get_initial_asset_price(self) -> torch.Tensor:
        return torch.ones(1, device=DEVICE) * self.config.initial_asset_price


class BlackScholesDiffusionCoefficient(Coefficient[BlackScholesCoefficientConfig]):

    def __init__(self, config: BlackScholesCoefficientConfig):
        super().__init__(config)
        self.sigma = nn.Parameter(
            torch.tensor(
                self.config.initializer.sigma,
                dtype=torch.float32,
                device=DEVICE,
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.eliminate_time_if_time_invariant(x)
        return (x * self.sigma)[:, :, None]

