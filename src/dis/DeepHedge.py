from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy._typing import NDArray
from torch import nn
from tqdm import trange

from src.config import DEVICE
from src.deep_hedging.StrategyNet import StrategyNet, StrategyNetConfig
from src.deep_hedging.objectives.HedgeObjective import HedgeObjective, Std
from src.dis.Derivative import Derivative, EuroCall
from src.gen.Generator import Generator
from src.gen.Heston import HestonParameterSet, HestonGeneratorConfig, HestonGenerator
from src.util.TimeUtil import UniformTimeDiscretization


@dataclass
class Masks:
    observable: Optional[NDArray] = None
    tradable: Optional[NDArray] = None
    payoff: Optional[NDArray] = None

    def ob(self, paths: torch.Tensor) -> torch.Tensor:
        return self._mask(paths, self.observable)

    def tr(self, paths: torch.Tensor) -> torch.Tensor:
        return self._mask(paths, self.tradable)

    def po(self, paths: torch.Tensor) -> torch.Tensor:
        return self._mask(paths, self.payoff)

    @staticmethod
    def _mask(paths: torch.Tensor, mask: Optional[NDArray]):
        if mask is None:
            return paths

        res = paths[:, :, mask]

        if len(res.shape) != 3:  # If reduced to 1-dim
            res = res[:, :, None]

        return res


class DeepHedge:

    def __init__(
            self,
            strategy: StrategyNet,
            derivative: Derivative,
            generator: Generator,
            objective: HedgeObjective,
            optimizer: Optional[torch.optim.Optimizer] = None,
            masks: Optional[Masks] = None,
    ):
        self.strategy = strategy
        self.derivative = derivative
        self.generator = generator
        self.objective = objective
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(self.strategy.parameters())
        self.masks = masks if masks is not None else Masks()

        self.dtype = torch.float32
        self.device = DEVICE
        self.td = self.generator.config.td

    def step(
            self,
            batch_size: int,
            noise: Optional[torch.Tensor] = None,
            initial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.objective(self.compute_pnl(batch_size, noise, initial))

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

    def compute_pnl(
            self,
            batch_size: int,
            noise: Optional[torch.Tensor] = None,
            initial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        paths = self.generator(batch_size, noise, initial)
        return self.compute_pnl_on_paths(paths)

    def compute_pnl_on_paths(self, paths: torch.Tensor) -> torch.Tensor:
        terminal_wealth = self.compute_wealth(paths)[:, -1]
        payoff = self.derivative.get_payoff_for_paths(self.masks.po(paths))
        return terminal_wealth - payoff

    def compute_wealth(self, paths: torch.Tensor) -> torch.Tensor:
        batch_size = paths.shape[0]
        times = torch.tensor(self.td.times, dtype=self.dtype, device=self.device)[None, :, None]
        observations = torch.cat((times.repeat(batch_size, 1, 1), self.masks.ob(paths)), dim=2)[:, :-1, :]
        flat_observations = observations.reshape(batch_size * self.td.number_of_time_steps, -1)

        actions = self.strategy(flat_observations).reshape(batch_size, self.td.number_of_time_steps, -1)

        wealth_increments = (torch.diff(self.masks.tr(paths), 1, 1) * actions).sum(dim=2)
        return torch.cat((torch.zeros((batch_size, 1)), torch.cumsum(wealth_increments, dim=1)), dim=1)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != "pde_optimizer":
                setattr(result, k, deepcopy(v, memo))

        result.pde_optimizer = self.optimizer.__class__(result.strategy.parameters())
        self.sync_lrs(self.optimizer, result.optimizer)

        return result

    @staticmethod
    def sync_lrs(opt_old, opt_new) -> None:
        for g_in_old, g_in_new in zip(opt_old.param_groups, opt_new.param_groups):
            g_in_new['lr'] = g_in_old['lr']


if __name__ == '__main__':
    pars = HestonParameterSet(3.0, 0.03, 0.2, -.8)
    tdis = UniformTimeDiscretization.from_bounds(0, .25, 18)
    conf = HestonGeneratorConfig(tdis, pars, True)

    gen = HestonGenerator(conf)
    net = StrategyNet(StrategyNetConfig(2, 2, 3, 64, output_activation=nn.ReLU()))
    eu_c = EuroCall(1.0)
    o = Std()
    m = Masks(np.array((True, True, False)), np.array((True, False, True)), np.array((True, False, False)))

    hedge = DeepHedge(net, eu_c, gen, o, masks=m)

    for _ in trange(1000):
        hedge.step(2 ** 8)

    p = gen(2 ** 8)
    plt.scatter(p[:, -1, 0].detach().numpy(), hedge.compute_wealth(p)[:, -1].detach().numpy())
    plt.show()
