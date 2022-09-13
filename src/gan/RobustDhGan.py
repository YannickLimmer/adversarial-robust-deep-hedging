from dataclasses import dataclass, asdict, fields
from typing import Dict, List, Optional, Tuple

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
            trainer_config: TrainerConfig,
    ):
        super().__init__(trainer_config)
        self.hedge_config = hedge_config
        self.gen_config = gen_config

        self._train_hedge = True
        self._train_generator = True

    def step(self, noise: torch.Tensor) -> RobustDhGanMetrics:
        hedge_loss = self.deep_hedge_step(noise)
        generation_loss, penalty = self.generator_step(noise)
        return RobustDhGanMetrics(hedge_loss, penalty, generation_loss)

    def deep_hedge_step(self, noise: torch.Tensor) -> Optional[torch.Tensor]:
        if self._train_hedge:
            hedge_loss = self.compute_hedge_loss(noise)
            self.update_deep_hedge(hedge_loss)
            return hedge_loss

    def update_deep_hedge(self, hedge_loss: torch.Tensor) -> None:
        self.hedge_config.deep_hedge.zero_grad()
        hedge_loss.backward()
        self.hedge_config.optimizer.step()
        self.hedge_config.scheduler.step()

    def compute_hedge_loss(self, noise: torch.Tensor) -> torch.Tensor:
        generated = self.gen_config.generator(noise)
        return self.compute_hedge_loss_from_generated(generated)

    def compute_hedge_loss_from_generated(self, generated: torch.Tensor) -> torch.Tensor:
        profit_and_loss = self.hedge_config.deep_hedge(self.hedge_config.generation_adapters(generated))
        return self.hedge_config.hedge_objective(profit_and_loss)

    def generator_step(self, noise) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._train_generator:
            generation_loss, penalty = self.compute_generation_loss(noise)
            self.update_generator(generation_loss)
            return generation_loss, penalty
        # Metrics can deal with None values.
        return None, None

    def update_generator(self, generation_loss: torch.Tensor) -> None:
        self.gen_config.generator.zero_grad()
        generation_loss.backward()
        self.gen_config.optimizer.step()
        self.gen_config.scheduler.step()

    def compute_generation_loss(self, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        generated = self.gen_config.generator(noise)
        updated_hedge_loss = self.compute_hedge_loss_from_generated(generated) if self._train_hedge else 0.0
        penalty = self.gen_config.penalizer(self.gen_config.penalization_adapters(generated))
        return (penalty - updated_hedge_loss), penalty

    def deactivate_generation_training(self) -> None:
        self._train_generator = False

    def activate_generation_training(self) -> None:
        self._train_generator = True

    def deactivate_hedge_training(self) -> None:
        self._train_hedge = False

    def activate_hedge_training(self) -> None:
        self._train_hedge = True
