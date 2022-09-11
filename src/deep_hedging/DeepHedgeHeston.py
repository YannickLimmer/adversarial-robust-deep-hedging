import torch

from src.deep_hedging.DeepHedge import DeepHedge
from src.deep_hedging.AbstractDeepHedge import DeepHedgeConfig


class DeepHedgeHestonConfig(DeepHedgeConfig):

    def __post_init__(self):
        if self.initial_information_value.shape[0] != 3:
            raise AttributeError(
                'Initial asset price must have three entries, initial asset price, initial volatility and initial '
                'volatility swap price.'
            )
        if self.strategy_config.dim_of_information_process != 2:
            raise AttributeError('Strategy requires 2 input dimensions.')


class DeepHedgeHeston(DeepHedge[DeepHedgeHestonConfig]):

    def _filter_information(self, information: torch.Tensor) -> torch.Tensor:
        return information[:, (0, 1, 2)]

    @staticmethod
    def _update_wealth(
            wealth: torch.Tensor,
            inputs: torch.Tensor,
            strategy_for_time_step: torch.Tensor,
            time_step_index: int,
    ) -> torch.Tensor:
        return wealth + torch.sum(strategy_for_time_step * inputs[:, time_step_index, (0, 2)], dim=1)
