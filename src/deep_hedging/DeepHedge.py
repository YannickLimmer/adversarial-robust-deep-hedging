from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.deep_hedging.StrategyNet import StrategyNetConfig, StrategyNet
from src.derivative.Derivative import Derivative


@dataclass
class DeepHedgeConfig:
    derivative: Derivative
    initial_asset_price: torch.Tensor
    strategy_config: StrategyNetConfig

    def __post_init__(self):
        if self.initial_asset_price.shape[0] != self.strategy_config.dimension_of_asset:
            raise AttributeError(
                'Initial asset price dimension does not coincide with asset price dimension of strategy specs.',
            )


class DeepHedge(nn.Module):

    def __init__(self, config: DeepHedgeConfig):
        super().__init__()
        self.config = config
        self.td = config.derivative.td

        self.strategy_layer = StrategyNet(self.config.strategy_config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        information = self.extract_initial_information(inputs)
        wealth = self.create_initial_wealth(inputs)

        strategy = []
        for time_step_index in self.td.indices:
            strategy.append(self.strategy_layer(information))
            information = self.update_information(information, inputs, time_step_index)
            wealth = self.update_wealth(wealth, inputs, strategy[-1], time_step_index)

        profit_and_loss = self.calculate_profit_and_loss(information, wealth)

        if self.training:
            return self._prepare_pnl_for_training(profit_and_loss)

        return torch.stack(strategy, dim=1)

    def extract_initial_information(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                self.td.times[0] * torch.ones_like(inputs[:, 0, 0:1]),
                self.config.initial_asset_price * torch.ones_like(inputs[:, 0, :]),
            ),
            dim=1,
        )

    # noinspection PyMethodMayBeStatic
    def create_initial_wealth(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((inputs.shape[0]))

    def update_information(self, information: torch.Tensor, inputs: torch.Tensor, time_step_index: int) -> torch.Tensor:
        return torch.cat(
            (
                self.td.times[time_step_index + 1] * torch.ones_like(inputs[:, 0, 0:1]),
                information[:, 1:] + inputs[:, time_step_index],
            ),
            dim=1,
        )

    @staticmethod
    def update_wealth(
            wealth: torch.Tensor,
            inputs: torch.Tensor,
            strategy_for_time_step: torch.Tensor,
            time_step_index: int,
    ) -> torch.Tensor:
        return wealth + torch.sum(strategy_for_time_step * inputs[:, time_step_index], dim=1)

    def calculate_profit_and_loss(self, information: torch.Tensor, wealth: torch.Tensor) -> torch.Tensor:
        derivative_payoff = self.config.derivative.payoff_for_terminal_asset_values(
            self.terminal_asset_price_from_information(information),
        )
        return self.config.derivative.price + wealth - derivative_payoff

    # noinspection PyMethodMayBeStatic
    def terminal_asset_price_from_information(self, information: torch.Tensor) -> torch.Tensor:
        return information[:, 1:]

    # noinspection PyMethodMayBeStatic
    def _prepare_pnl_for_training(self, profit_and_loss: torch.Tensor) -> torch.Tensor:
        return profit_and_loss
