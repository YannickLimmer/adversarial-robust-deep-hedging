from dataclasses import dataclass, asdict, fields
from typing import Dict, List, Optional, Tuple, Any, OrderedDict

import numpy as np
import torch
from numpy.typing import NDArray

from src.deep_hedging.DeepHedge import DeepHedge
from src.deep_hedging.DeepHedgeTrainer import DeepHedgeTrainerConfig
from src.generator.EulerGenerator import EulerGenerator
from src.generator.SdeGeneratorTrainer import SdeGeneratorTrainerConfig
from src.util.torch_util.TrainingUtil import Trainer, TrainerConfig, Metrics, PbarOption


@dataclass
class RobustDhGanMetrics(Metrics):

    hedge_loss: torch.Tensor = None
    penalty: torch.Tensor = None
    generation_loss: torch.Tensor = None

    def create_print_dict(self) -> Dict[str, float]:
        return {k: v.item() if v is not None else float('nan') for k, v in self.__dict__.items()}

    @property
    def loss(self) -> NDArray:
        return np.array([v.item() if v is not None else float('nan') for v in self.__dict__.values()])

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
            dh: DeepHedge,
            hedge_config: DeepHedgeTrainerConfig,
            gen: EulerGenerator,
            gen_config: SdeGeneratorTrainerConfig,
            trainer_config: TrainerConfig,
    ):
        super().__init__(trainer_config)
        self.dh = dh
        self.dh_optimizer = torch.optim.Adam(self.dh.parameters())
        self.dh_config = hedge_config
        self.dh_scheduler = self.dh_config.scheduler_from_optimizer(self.dh_optimizer)

        self.gen = gen
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters())
        self.gen_config = gen_config
        self.gen_scheduler = self.gen_config.scheduler_from_optimizer(self.gen_optimizer)

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
        self.dh.zero_grad()
        hedge_loss.backward()
        self.dh_optimizer.step()
        # self.dh_scheduler.step()

    def compute_hedge_loss(self, noise: torch.Tensor) -> torch.Tensor:
        generated = self.gen(noise)
        return self.compute_hedge_loss_from_generated(generated)

    def compute_hedge_loss_from_generated(self, generated: torch.Tensor) -> torch.Tensor:
        profit_and_loss = self.dh(self.dh_config.generation_adapters(generated))
        return self.dh_config.hedge_objective(profit_and_loss)

    def generator_step(self, noise) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._train_generator:
            generation_loss, penalty = self.compute_generation_loss(noise)
            self.update_generator(generation_loss)
            return generation_loss, penalty
        # Metrics can deal with None values.
        return None, None

    def update_generator(self, generation_loss: torch.Tensor) -> None:
        self.gen.zero_grad()
        generation_loss.backward()
        self.gen_optimizer.step()
        # self.gen_scheduler.step()

    def compute_generation_loss(self, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        generated = self.gen(noise)
        updated_hedge_loss = self.compute_hedge_loss_from_generated(generated) if self._train_hedge else 0.0
        penalty = self.gen_config.penalizer(self.gen_config.penalization_adapters(generated))
        return (penalty - updated_hedge_loss), penalty

    def load_module_from_state_dict(self, f: Any) -> None:
        self.load_state_dict(torch.load(f))

    def get_weights_from_load_and_fitting_procedure(
            self,
            f: Any,
            batch_sizes: List[int],
            pbar_option: PbarOption = PbarOption.NO_BAR,
            pretrained: Optional[Any] = None,
            loss_curve_address: Optional[Any] = None,
            parameter_tracking_address: Optional[Any] = None,
    ) -> OrderedDict[str, torch.Tensor]:
        self.load_or_fit(f, batch_sizes, pbar_option, pretrained, loss_curve_address, parameter_tracking_address)
        return self.state_dict()

    @property
    def tracked_parameters(self) -> Dict[str, NDArray]:
        return {k: v.clone().detach().numpy() for k, v in self.gen.named_parameters()}

    def deactivate_generation_training(self) -> None:
        self._train_generator = False

    def activate_generation_training(self) -> None:
        self._train_generator = True

    def deactivate_hedge_training(self) -> None:
        self._train_hedge = False

    def activate_hedge_training(self) -> None:
        self._train_hedge = True
