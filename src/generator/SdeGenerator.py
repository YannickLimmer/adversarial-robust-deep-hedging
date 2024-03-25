from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from typing import Callable, TypeVar, Generic

import torch

from src.util.TimeUtil import TimeDiscretization


@dataclass
class GeneratorConfig:
    td: TimeDiscretization
    initial_asset_price: Callable[[], torch.Tensor]


_GeneratorConfig = TypeVar('_GeneratorConfig', bound=GeneratorConfig)


class SdeGenerator(torch.nn.Module, Generic[_GeneratorConfig], metaclass=ABCMeta):

    def __init__(self, generator_config: GeneratorConfig):
        super().__init__()
        self.config = generator_config

    @property
    @abstractmethod
    def initial_asset_price(self) -> Callable[[], torch.Tensor]:
        pass

    @abstractmethod
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        pass
