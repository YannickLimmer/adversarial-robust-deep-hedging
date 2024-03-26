from _py_abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass

import torch


class Derivative(metaclass=ABCMeta):

    @abstractmethod
    def get_payoff_for_paths(self, paths: torch.Tensor) -> torch.Tensor:
        pass


@dataclass
class EuroCall(Derivative):

    strike: float

    def get_payoff_for_paths(self, paths: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu((paths[:, -1, :] - self.strike).sum(dim=1))


@dataclass
class MaxEuroCall(Derivative):

    strike: float

    def get_payoff_for_paths(self, paths: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(torch.max(paths[:, -1, :], dim=1)[0] - self.strike)
