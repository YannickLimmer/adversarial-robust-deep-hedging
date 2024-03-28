from abc import abstractmethod, ABCMeta
from dataclasses import dataclass

import numpy as np
import torch
from typing import Optional, List, Type
from functools import wraps

from matplotlib import pyplot as plt

from src.gen.Generator import Generator, GeneratorConfig_
from src.gen.Heston import HestonParameterSet, HestonGeneratorConfig, HestonGenerator
from src.util.TimeUtil import UniformTimeDiscretization


class Feature(metaclass=ABCMeta):

    def __call__(self, cls: Type[Generator]) -> Type[Generator]:

        feature_from_paths = self.feature_from_paths

        # @wraps(cls)
        class DecoratedClass(cls):
            def forward(
                    self,
                    batch_size: int,
                    noise: Optional[torch.Tensor] = None,
                    initial: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                paths = super().forward(batch_size, initial, noise)
                custom_feature = feature_from_paths(paths, self.config)
                return torch.cat((paths, custom_feature), dim=2)

        return DecoratedClass

    @abstractmethod
    def feature_from_paths(self, paths: torch.Tensor, config: GeneratorConfig_) -> torch.Tensor:
        pass


@dataclass
class CatRunningAverage(Feature):

    dim: int

    def feature_from_paths(self, paths: torch.Tensor, config: GeneratorConfig_) -> torch.Tensor:
        return (torch.cumsum(paths[:, :, self.dim], dim=1) / (1 + torch.arange(paths.shape[1])))[:, :, None]


@dataclass
class CatRunningMin(Feature):

    dim: int

    def feature_from_paths(self, paths: torch.Tensor, config: GeneratorConfig_) -> torch.Tensor:
        return torch.cummin(paths[:, :, self.dim], dim=1).values[:, :, None]


@dataclass
class CatRunningMax(Feature):

    dim: int

    def feature_from_paths(self, paths: torch.Tensor, config: GeneratorConfig_) -> torch.Tensor:
        return torch.cummax(paths[:, :, self.dim], dim=1).values[:, :, None]


@dataclass
class CatBarrier(Feature):

    barrier: float
    dim: int
    above: Optional[bool] = None

    def feature_from_paths(self, paths: torch.Tensor, config: GeneratorConfig_) -> torch.Tensor:
        if self.barrier_is_above_initial_price(paths[:, 0, self.dim]):
            return torch.bitwise_not(torch.cumprod(paths[:, :, self.dim] < self.barrier, dim=1))[:, :, None]
        return torch.bitwise_not(torch.cumprod(paths[:, :, self.dim] > self.barrier, dim=1))[:, :, None]

    def barrier_is_above_initial_price(self, initial: torch.Tensor) -> bool:
        if self.above is not None:
            return self.above
        if torch.all(initial <= self.barrier):
            return True
        if torch.all(initial > self.barrier):
            return False
        raise AttributeError("Ambiguous Barrier definition, please provide `above` attribute.")


@dataclass
class CatStartStrikes(Feature):

    time_indices: List[int] # Starts with 0 and ends with the last time index
    dim: int

    def __post_init__(self):
        self.time_indices_arr = np.array(self.time_indices)
        self.repeats = np.concatenate((np.diff(self.time_indices_arr), np.ones(1)))

    def feature_from_paths(self, paths: torch.Tensor, config: GeneratorConfig_) -> torch.Tensor:
        repeats_tensor = torch.tensor(self.repeats, dtype=torch.int64, device=paths.device)
        return torch.repeat_interleave(paths[:, self.time_indices_arr, self.dim], repeats=repeats_tensor, dim=1)[:, :, None]
    

if __name__ == '__main__':
    pars = HestonParameterSet(3.0, 0.03, 0.2, -.8)
    tdis = UniformTimeDiscretization.from_bounds(0, 1, 60)
    conf = HestonGeneratorConfig(tdis, pars, False)
    cat_running_average = CatRunningAverage(0)
    cat_barrier = CatBarrier(1.15, dim=0)
    cat_running_min = CatRunningMin(1)
    cat_running_max = CatRunningMax(1)
    cat_start1 = CatStartStrikes([0, 30, 60], 0)
    cat_start2 = CatStartStrikes([0, 20, 40, 60], 1)
    gen = cat_start2(cat_start1(cat_running_max(cat_running_min(cat_barrier(cat_running_average(
        HestonGenerator
    ))))))(conf)
    p = gen(100).detach()
    print(gen.default_initial)

    fig, ax = plt.subplots(4, 2, figsize=(8, 16))
    ax[0, 0].plot(tdis.times, p.numpy()[:, :, 0].T)
    ax[0, 1].plot(tdis.times, p.numpy()[:, :, 1].T)
    ax[1, 0].plot(tdis.times, p.numpy()[:, :, 2].T)
    ax[2, 0].plot(tdis.times, p.numpy()[:, :, 3].T)
    ax[1, 1].plot(tdis.times, p.numpy()[:, :, 4].T)
    ax[2, 1].plot(tdis.times, p.numpy()[:, :, 5].T)
    ax[3, 0].plot(tdis.times, p.numpy()[:, :, 6].T)
    ax[3, 1].plot(tdis.times, p.numpy()[:, :, 7].T)
    plt.show()
