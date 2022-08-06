from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Generic, TypeVar

import torch
import torch.nn


@dataclass
class CoefficientConfig:
    dimension_of_process: Optional[int] = field(default=None, init=False)
    time_invariant: bool = field(default=False, init=False)


_CoefficientConfig = TypeVar('_CoefficientConfig', bound=CoefficientConfig)


class Coefficient(torch.nn.Module, Generic[_CoefficientConfig], metaclass=ABCMeta):

    def __init__(self, config: _CoefficientConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def eliminate_time_if_time_invariant(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 1:] if self.config.time_invariant else x


_Drift_Coefficient = TypeVar('_Drift_Coefficient', bound=Coefficient)
_Diffusion_Coefficient = TypeVar('_Diffusion_Coefficient', bound=Coefficient)

