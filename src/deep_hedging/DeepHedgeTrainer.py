from dataclasses import dataclass
from typing import Dict, Any, List, OrderedDict, Optional

import torch
from numpy.typing import NDArray
from torch.optim.lr_scheduler import StepLR

from src.deep_hedging.DeepHedge import DeepHedge
from src.deep_hedging.objectives.HedgeObjective import HedgeObjective
from src.util.torch_util.AdapterUtil import AdapterList
from src.util.torch_util.TrainingUtil import Trainer, TrainerConfig, Metrics, PbarOption


@dataclass
class DeepHedgeTrainerConfig:
    hedge_objective: HedgeObjective
    generation_adapters: AdapterList = None
    scheduler_step_size: int = 100
    scheduler_gamma: float = 0.9

    def __post_init__(self):
        self.generation_adapters = self.generation_adapters if self.generation_adapters else AdapterList()

    def scheduler_from_optimizer(self, optimizer: torch.optim.Optimizer) -> StepLR:
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )


@dataclass
class HedgeMetrics(Metrics):

    profit_and_loss: torch.Tensor = None
    hedge_loss: torch.Tensor = None

    @property
    def loss(self) -> NDArray:
        return self.hedge_loss.item()

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

    def __init__(self, dh: DeepHedge, config: DeepHedgeTrainerConfig, trainer_config: TrainerConfig):
        super().__init__(trainer_config)
        self.dh = dh
        self.optimizer = torch.optim.Adam(self.dh.parameters())
        self.config = config
        self.scheduler = self.config.scheduler_from_optimizer(self.optimizer)

    def step(self, inputs: torch.Tensor) -> HedgeMetrics:
        m = HedgeMetrics()

        # Compute profit and loss and hedge loss
        m.profit_and_loss = self.dh(inputs)
        m.hedge_loss = self.config.hedge_objective(m.profit_and_loss)

        # Compute gradient and update weights
        self.dh.zero_grad()
        m.hedge_loss.backward()
        self.optimizer.step()
        # self.scheduler.step()

        return m

    def get_weights_from_load_and_fitting_procedure(
            self,
            f: Any,
            batch_sizes: List[int],
            pbar_option: PbarOption = PbarOption.NO_BAR,
            pretrained: Optional[Any] = None,
            loss_curve_address: Optional[Any] = None,
    ) -> OrderedDict[str, torch.Tensor]:
        self.load_or_fit(f, batch_sizes, pbar_option, pretrained, loss_curve_address)
        return self.dh.state_dict()

    def load_module_from_state_dict(self, f: Any) -> None:
        self.dh.load_state_dict(torch.load(f))

    @property
    def tracked_parameters(self) -> Dict[str, NDArray]:
        return {}

