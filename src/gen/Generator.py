import math
from abc import abstractmethod, ABCMeta
from builtins import NotImplemented
from dataclasses import dataclass
from typing import Generic, TypeVar, Union, Optional

import numpy as np
import torch
from torch import nn

from src.config import DEVICE
from src.util.TimeUtil import TimeDiscretization


@dataclass
class GeneratorConfig:
    td: TimeDiscretization
    process_dim: int


GeneratorConfig_ = TypeVar("GeneratorConfig_", bound=GeneratorConfig)


class Generator(nn.Module, Generic[GeneratorConfig_], metaclass=ABCMeta):

    def __init__(self, config: GeneratorConfig_, rng: Union[None, int, np.random.Generator] = None):
        super().__init__()
        self.config = config
        self.rng = np.random.default_rng(rng)

        self.dtype = torch.float32
        self.device = DEVICE

        self._register__()

    def _register__(self):
        pass

    def forward(
            self,
            batch_size: int,
            noise: Optional[torch.Tensor] = None,
            initial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        initial = self.get_initial(batch_size, initial)
        return self._forward(batch_size, initial, noise)

    def get_initial(self, batch_size: int, initial: Optional[torch.Tensor]) -> torch.Tensor:
        initial = self.default_initial if initial is None else initial
        if len(initial.shape) == 1:
            initial = initial.repeat(batch_size, 1)
        return initial

    @property
    def default_initial(self) -> torch.Tensor:
        raise NotImplemented("Default initial value is not specified for this class")

    @abstractmethod
    def _forward(
            self,
            batch_size: int,
            initial: torch.Tensor,
            noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def verify_shape(arr, shape):
        if arr.shape != shape:
            raise AttributeError(f"Shape mismatch, expected {shape}, but noise is of shape {arr.shape}")


@dataclass
class DiffusionGeneratorConfig(GeneratorConfig):
    noise_dim: int


class EulerGenerator(Generator[DiffusionGeneratorConfig], Generic[GeneratorConfig_], metaclass=ABCMeta):

    def _forward(
            self,
            batch_size: int,
            initial: torch.Tensor,
            noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = self.get_noise(batch_size, noise)
        return self.paths_from_noise(noise, initial)

    def get_noise(self, batch_size: int, noise: Optional[torch.Tensor]) -> torch.Tensor:
        shape = (batch_size, self.config.td.number_of_time_steps, self.config.noise_dim)
        if noise is not None:
            self.verify_shape(noise, shape)
            return noise

        return torch.tensor(self.rng.normal(0.0, 1.0, size=shape), dtype=self.dtype, device=self.device)

    def paths_from_noise(self, noise: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
        process = [initial]
        for i, t in enumerate(self.config.td.time_steps):
            dt = self.config.td.time_step_increments[i]
            drift_increment = self.drift(t, process[-1]) * dt
            # Here, `diffusion` is of size (n, d, d,), `noise` of size (n, d, 1,), resulting in a multiplication of a
            # (d, d,) matrix with a (d,) column vector for every of the n entries.
            stoch_increment = (self.diffusion(t, process[-1]) @ noise[:, i, :, None])[:, :, 0] * math.sqrt(dt)
            process.append(process[-1] + drift_increment + stoch_increment)
        return torch.stack(process, dim=1)

    @abstractmethod
    def drift(self, t: float, prev: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def diffusion(self, t: float, prev: torch.Tensor) -> torch.Tensor:
        pass
