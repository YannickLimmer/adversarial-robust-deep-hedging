from dataclasses import dataclass

import torch

from src.util.torch_util.AdapterUtil import AdapterList


@dataclass
class SdeGeneratorTrainerConfig:
    generator: torch.nn.Module
    penalizer: torch.nn.Module
    optimizer: torch.optim.Optimizer
    penalization_adapters: AdapterList = AdapterList()

    def __post_init__(self):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

    def reset_scheduler(self) -> None:
        self.__post_init__()
