import numpy as np
import torch

from src.derivative.Derivative import Derivative
from src.util.TimeUtil import TimeDiscretization


class EuropeanCallOption(Derivative):

    def __init__(self, strike: float, time_discretization: TimeDiscretization, price: float = None):
        super().__init__(time_discretization, price)
        self.strike = strike

    def payoff_for_terminal_asset_values(self, terminal_values: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.maximum(terminal_values - self.strike, torch.zeros_like(terminal_values)), dim=1)
