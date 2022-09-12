from dataclasses import dataclass, asdict, fields
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
        return {k: v.item() if v is not None else float('nan') for k, v in asdict(self).items()}

    @classmethod
    def summarize_metrics_list_to_metrics(cls, metric_list: List['RobustDhGanMetrics']) -> 'RobustDhGanMetrics':
        return cls(
            **{f.name: None if getattr(metric_list[0], f.name) is None else torch.mean(
                torch.tensor([getattr(metr, f.name) for metr in metric_list])
            ) for f in fields(cls)},
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
        m = RobustDhGanMetrics()

        # Update Deep Hedge
        if self._train_hedge:
            # Generate and calculate hedge loss
            generated = self.gen_config.generator(noise)
            profit_and_loss = self.hedge_config.deep_hedge(self.hedge_config.generation_adapters(generated))
            m.hedge_loss = self.hedge_config.hedge_objective(profit_and_loss)

            # Compute gradients and update weights
            self.hedge_config.deep_hedge.zero_grad()
            m.hedge_loss.backward()
            self.hedge_config.optimizer.step()
            self.hedge_config.scheduler.step()

        # Update Generator
        if self._train_generator:
            # Generate and calculate penalized loss
            generated = self.gen_config.generator(noise)
            if self._train_hedge:
                profit_and_loss = self.hedge_config.deep_hedge(self.hedge_config.generation_adapters(generated))
                m.hedge_loss = self.hedge_config.hedge_objective(profit_and_loss)
            m.penalty = self.gen_config.penalizer(self.gen_config.penalization_adapters(generated))
            m.generation_loss = m.penalty - (0.0 if not self._train_hedge else m.hedge_loss)

            # Compute gradients and update weights
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
