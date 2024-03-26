from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from numpy._typing import NDArray

from src.deep_hedging.StrategyNet import StrategyNet
from src.deep_hedging.objectives.HedgeObjective import HedgeObjective
from src.dis.Derivative import Derivative
from src.gen.Generator import Generator


@dataclass
class Masks:
    observable: Optional[NDArray] = None
    tradable: Optional[NDArray] = None
    payoff: Optional[NDArray] = None
    process_dim: Optional[int] = None

    def __post_init__(self):
        mask_list = [self.observable, self.tradable, self.payoff]
        non_none_maks = [len(m) for m in mask_list if m is not None]
        if self.process_dim is None:
            self.process_dim = non_none_maks[0]

        if self.observable is None:
            self.observable = np.ones(self.process_dim, dtype=bool)
        if self.tradable is None:
            self.tradable = np.ones(self.process_dim, dtype=bool)
        if self.payoff is None:
            self.payoff = np.ones(self.process_dim, dtype=bool)


class DeepHedge:

    def __init__(
            self,
            strategy: StrategyNet,
            derivative: Derivative,
            generator: Generator,
            objective: HedgeObjective,
            optimizer: Optional[torch.optim.optimizer] = None,
            masks: Optional[Masks] = None,
    ):
        self.strategy = strategy
        self.derivative = derivative
        self.generator = generator
        self.objective = objective
        self.optimizer = optimizer if not optimizer else torch.optim.Adam(self.strategy.parameters())
        self.masks = masks if masks is not None else Masks(process_dim=generator.config.process_dim)

    def step(
            self,
            batch_size: int,
            noise: Optional[torch.Tensor] = None,
            initial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        paths = self.generator(batch_size, noise, initial)
        loss = self.objective(self.compute_pnl(paths))

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

    def compute_pnl(self, paths):
        terminal_wealth = self.compute_wealth(paths)[:, -1]
        payoff = self.derivative.get_payoff_for_paths(paths[:, :, self.masks.payoff])
        return terminal_wealth - payoff

    def compute_wealth(self, paths: torch.Tensor) -> torch.Tensor:
        observable_paths_with_time = torch.cat((paths[:, :, self.masks.observable]))

