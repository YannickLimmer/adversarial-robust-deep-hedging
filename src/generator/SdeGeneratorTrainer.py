from dataclasses import dataclass
from typing import List, Dict, Any, Optional, OrderedDict

import torch
from numpy.typing import NDArray
from torch.optim.lr_scheduler import StepLR

from src.generator.SdeGenerator import SdeGenerator
from src.penalty.Metric import Metric
from src.util.torch_util.AdapterUtil import AdapterList
from src.util.torch_util.TrainingUtil import Trainer, Metrics, _Metrics, TrainerConfig, PbarOption


@dataclass
class SdeGeneratorTrainerConfig:
    penalizer: Metric
    penalization_adapters: AdapterList = None
    scheduler_step_size: int = 100
    scheduler_gamma: float = 0.9

    def __post_init__(self):
        self.penalization_adapters = self.penalization_adapters if self.penalization_adapters else AdapterList()

    def scheduler_from_optimizer(self, optimizer: torch.optim.Optimizer) -> StepLR:
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )


@dataclass
class SdeGeneratorMetrics(Metrics):

    generation_loss: torch.Tensor = None

    def create_print_dict(self) -> Dict[str, float]:
        return {'G-Loss': self.generation_loss.item() if self.generation_loss is not None else float('nan')}

    @property
    def loss(self) -> NDArray:
        return self.generation_loss.item()

    @classmethod
    def summarize_metrics_list_to_metrics(cls, metrics_list: List['SdeGeneratorMetrics']) -> 'SdeGeneratorMetrics':
        return cls(
            generation_loss = None if metrics_list[0].generation_loss is None else torch.mean(
                torch.tensor([metr.generation_loss for metr in metrics_list])
            ),
        )


class SdeGeneratorTrainer(Trainer[SdeGeneratorMetrics]):

    def __init__(
            self,
            gen: SdeGenerator,
            gen_config: SdeGeneratorTrainerConfig,
            trainer_config: TrainerConfig,
    ):
        super().__init__(trainer_config)

        self.gen = gen
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters())
        self.gen_config = gen_config
        self.gen_scheduler = self.gen_config.scheduler_from_optimizer(self.gen_optimizer)

    def step(self, noise: torch.Tensor) -> _Metrics:
        generated = self.gen(noise)
        generation_loss = self.gen_config.penalizer(self.gen_config.penalization_adapters(generated))

        self.gen.zero_grad()
        generation_loss.backward()
        self.gen_optimizer.step()
        # self.gen_scheduler.step()

        return SdeGeneratorMetrics(generation_loss)

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
        return self.gen.state_dict()

    def load_module_from_state_dict(self, f: Any) -> None:
        self.gen.load_state_dict(torch.load(f))

    @property
    def tracked_parameters(self) -> Dict[str, NDArray]:
        return {k: v.clone().detach().numpy() for k, v in self.named_parameters()}
