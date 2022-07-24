from abc import ABCMeta, abstractmethod
from typing import Optional

import torch

from src.util.TimeUtil import TimeDiscretization


class Derivative(metaclass=ABCMeta):

    def __init__(self, time_discretization: TimeDiscretization, price: Optional[float] = None):
        self.td = time_discretization
        self._price = price

    @property
    def price(self) -> float:
        if self._price is None:
            raise ValueError('Price of the derivative is not specified yet.')
        # noinspection PyTypeChecker
        return self._price

    @abstractmethod
    def payoff_for_terminal_asset_values(self, terminal_values: torch.Tensor) -> torch.Tensor:
        pass

    def set_price(self, price: float) -> None:
        self._price = price

    def verify_that_price_is_set(self) -> None:
        if self._price is None:
            raise AttributeError("The price of the derivative is not specified.")
