import functools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from src.config import DEVICE
from src.deep_hedging.DeepHedge import DeepHedge
from src.deep_hedging.objectives.HedgeObjective import HedgeObjective
from src.penalty.Metric import Metric, MetricConfig
from src.util.TimeUtil import TimeDiscretization
from src.util.torch_util.AdapterUtil import ConvertToIncrements


@dataclass
class VolatilityComparisonConfig(MetricConfig):
    td: TimeDiscretization

    def __post_init__(self):
        self.time_root = torch.as_tensor(np.sqrt(self.td.time_step_increments), dtype=torch.float32, device=DEVICE)


class CompareVolatility(Metric[VolatilityComparisonConfig]):

    @property
    @functools.lru_cache()
    def original_volatility(self) -> torch.Tensor:
        return self.to_volatility(self.original)

    def forward(self, generated: torch.Tensor) -> torch.Tensor:
        generated_volatility = self.to_volatility(generated)
        return self.transform(
            torch.abs(self.original_volatility ** 2 - generated_volatility ** 2),
        )

    def to_volatility(self, arr: torch.Tensor) -> torch.Tensor:
        log_incr = torch.diff(torch.log(arr), 1, 1)
        return torch.std((log_incr - torch.mean(log_incr, dim=0)) / self.config.time_root)


@dataclass
class DynamicVolatilityComparisonConfig(VolatilityComparisonConfig):
    hedge_objective: HedgeObjective


class CompareVolatilityDynamically(Metric[DynamicVolatilityComparisonConfig]):

    def __init__(
            self,
            original: torch.Tensor,
            dh: DeepHedge,
            metric_config: DynamicVolatilityComparisonConfig,
            transform: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__(original, metric_config, transform=transform)
        self.dh = dh
        self.converter = ConvertToIncrements()

    def forward(self, generated: torch.Tensor) -> torch.Tensor:
        realized_vol = self.to_volatility(generated)
        hedge_objective = self.config.hedge_objective(self.dh(self.converter(generated)))
        return self.transform(hedge_objective * (self.original_volatility - realized_vol) ** 2)

    def to_volatility(self, arr: torch.Tensor) -> torch.Tensor:
        log_incr = torch.diff(torch.log(arr), 1, 1)
        return torch.std((log_incr - torch.mean(log_incr, dim=0)) / self.config.time_root)

    @property
    @functools.lru_cache()
    def original_volatility(self) -> torch.Tensor:
        return self.to_volatility(self.original)
