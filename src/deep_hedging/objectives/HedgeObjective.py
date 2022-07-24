from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import torch

from src.util.LossUtil import entropy, mean_variance, conditional_value_at_risk, negative_standard_deviation


@dataclass
class HedgeObjective(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    def __call__(self, profit_and_loss: torch.Tensor) -> torch.Tensor:
        return - self._call(profit_and_loss)

    @abstractmethod
    def _call(self, profit_and_loss: torch.Tensor) -> torch.Tensor:
        pass


@dataclass
class Entropy(HedgeObjective):

    risk_aversion: float

    def _call(self, profit_and_loss: torch.Tensor) -> torch.Tensor:
        return entropy(profit_and_loss, self.risk_aversion, dim=0)


@dataclass
class MeanVariance(HedgeObjective):

    risk_aversion: float

    def _call(self, profit_and_loss: torch.Tensor) -> torch.Tensor:
        return mean_variance(profit_and_loss, self.risk_aversion, dim=0)


@dataclass
class CVAR(HedgeObjective):

    considered_fraction: float

    def _call(self, profit_and_loss: torch.Tensor) -> torch.Tensor:
        return conditional_value_at_risk(profit_and_loss, self.considered_fraction, dim=0)

@dataclass
class Std(HedgeObjective):

    def _call(self, profit_and_loss: torch.Tensor) -> torch.Tensor:
        return negative_standard_deviation(profit_and_loss, dim=0)

