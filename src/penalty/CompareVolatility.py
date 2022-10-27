import functools
from dataclasses import dataclass

import numpy as np
import torch

from src.config import DEVICE
from src.penalty.Metric import Metric, MetricConfig
from src.util.TimeUtil import TimeDiscretization


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
