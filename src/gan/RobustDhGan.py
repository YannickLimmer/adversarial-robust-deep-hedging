from dataclasses import dataclass
from typing import Dict, List

import torch

from src.deep_hedging.DeepHedgeTrainer import DeepHedgeTrainerConfig
from src.generator.SdeGeneratorTrainer import SdeGeneratorTrainerConfig
from src.util.torch_util.TrainingUtil import Trainer, TrainerConfig, Metrics


@dataclass
class RobustDhGanMetrics(Metrics):

    hedge_loss: torch.Tensor = None
    penalty: torch.Tensor = None
    generation_loss: torch.Tensor = None

    def create_print_dict(self) -> Dict[str, float]:
        return {
            'H-Loss': self.hedge_loss.item() if self.hedge_loss is not None else float('nan'),
            'G-Loss': self.generation_loss.item() if self.generation_loss is not None else float('nan'),
            'Penlty': self.penalty.item() if self.penalty is not None else float('nan'),
        }

    @classmethod
    def average_metric(cls, metric_list: List['RobustDhGanMetrics']) -> 'RobustDhGanMetrics':
        return cls(
            hedge_loss=None if metric_list[0].hedge_loss is None else torch.mean(
                torch.tensor([metr.hedge_loss for metr in metric_list])
            ),
            penalty=None if metric_list[0].penalty is None else torch.mean(
                torch.tensor([metr.penalty for metr in metric_list])
            ),
            generation_loss=None if metric_list[0].generation_loss is None else torch.mean(
                torch.tensor([metr.generation_loss for metr in metric_list])
            ),
        )


class RobustDhGan(Trainer[RobustDhGanMetrics]):

    def __init__(
            self,
            hedge_config: DeepHedgeTrainerConfig,
            gen_config: SdeGeneratorTrainerConfig,
            trainer_config: TrainerConfig
    ):
        super().__init__(trainer_config)
        self.hedge_config = hedge_config
        self.gen_config = gen_config

        self._train_hedge = True
        self._train_generator = True

    def step(self, noise: torch.Tensor) -> RobustDhGanMetrics:
        metrics = []
        for batch in self.batch_inputs(noise):
            metrics.append(self.step_on_batch(batch))
        return RobustDhGanMetrics.average_metric(metrics)

    def step_on_batch(self, batch):
        m = RobustDhGanMetrics()
        # Update Deep Hedge
        if self._train_hedge:
            generated = self.gen_config.generator(batch)
            profit_and_loss = self.hedge_config.deep_hedge(self.hedge_config.generation_adapters(generated))
            m.hedge_loss = self.hedge_config.hedge_objective(profit_and_loss)
            self.hedge_config.deep_hedge.zero_grad()
            m.hedge_loss.backward()
            self.hedge_config.optimizer.step()
            self.hedge_config.scheduler.step()
        # Update Generator
        if self._train_generator:
            generated = self.gen_config.generator(batch)
            if self._train_hedge:
                profit_and_loss = self.hedge_config.deep_hedge(self.hedge_config.generation_adapters(generated))
                m.hedge_loss = self.hedge_config.hedge_objective(profit_and_loss)
            m.penalty = self.gen_config.penalizer(self.gen_config.penalization_adapters(generated))
            m.generation_loss = m.penalty - (0.0 if not self._train_hedge else m.hedge_loss)
            self.gen_config.generator.zero_grad()
            m.generation_loss.backward()
            self.gen_config.optimizer.step()
            self.gen_config.scheduler.step()
        return m

    def deactivate_generation_training(self) -> None:
        self._train_generator = False

    def activate_generation_training(self) -> None:
        self._train_generator = True

    def deactivate_hedge_training(self) -> None:
        self._train_hedge = False

    def activate_hedge_training(self) -> None:
        self._train_hedge = True
