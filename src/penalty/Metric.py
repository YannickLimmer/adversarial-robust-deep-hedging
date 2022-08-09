from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Callable, Generic

import numpy as np
import torch
from torch.nn import Module

from src.config import DEVICE


@dataclass
class MetricConfig:
    pass


_MetricConfig = TypeVar('_MetricConfig', bound=MetricConfig)


class Metric(Module, Generic[_MetricConfig], metaclass=ABCMeta):

    def __init__(
            self,
            original: torch.Tensor,
            metric_config: _MetricConfig,
            transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    ):
        super().__init__()
        self.original = original
        self.config = metric_config
        self.transform = transform

    @abstractmethod
    def forward(self, generated):
        pass

    def sample_indices(self, batch_size: int):
        return torch.from_numpy(np.random.choice(self.original.shape[0], size=batch_size, replace=True)).to(DEVICE)
