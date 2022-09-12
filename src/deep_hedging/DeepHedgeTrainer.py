from dataclasses import dataclass
from typing import Dict, Any, List

import torch

from src.deep_hedging.DeepHedge import DeepHedge
from src.deep_hedging.objectives.HedgeObjective import HedgeObjective
from src.util.torch_util.AdapterUtil import AdapterList
from src.util.torch_util.TrainingUtil import Trainer, TrainerConfig, Metrics


@dataclass
class DeepHedgeTrainerConfig:
    deep_hedge: DeepHedge
    hedge_objective: HedgeObjective
    optimizer: torch.optim.Optimizer
    generation_adapters: AdapterList = AdapterList()

    def __post_init__(self):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

    def reset_scheduler(self) -> None:
        self.__post_init__()


@dataclass
class HedgeMetrics(Metrics):

    profit_and_loss: torch.Tensor = None
    hedge_loss: torch.Tensor = None

    def create_print_dict(self) -> Dict[str, float]:
        return {'H-Loss': self.hedge_loss.item() if self.hedge_loss is not None else float('nan')}

    @classmethod
    def summarize_metrics_list_to_metrics(cls, metrics_list: List['HedgeMetrics']) -> 'HedgeMetrics':
        return cls(
            profit_and_loss=None if metrics_list[0].profit_and_loss is None else torch.cat(
                [metr.profit_and_loss for metr in metrics_list], dim=0,
            ),
            hedge_loss=None if metrics_list[0].hedge_loss is None else torch.mean(
                torch.tensor([metr.hedge_loss for metr in metrics_list])
            ),
        )


class DeepHedgeTrainer(Trainer[HedgeMetrics]):

    def __init__(self, config: DeepHedgeTrainerConfig, trainer_config: TrainerConfig):
        super().__init__(trainer_config)
        self.config = config

    def step(self, inputs: torch.Tensor) -> HedgeMetrics:
        m = HedgeMetrics()

        # Compute profit and loss and hedge loss
        m.profit_and_loss = self.config.deep_hedge(inputs)
        m.hedge_loss = self.config.hedge_objective(m.profit_and_loss)

        # Compute gradient and update weights
        self.config.deep_hedge.zero_grad()
        m.hedge_loss.backward()
        self.config.optimizer.step()
        self.config.scheduler.step()

        return m


