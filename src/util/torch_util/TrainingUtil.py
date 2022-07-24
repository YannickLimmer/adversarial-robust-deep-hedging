from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from typing import List, Callable, Optional, Dict, TypeVar, Generic

import torch


@dataclass
class TrainerConfig:
    batch_size: int
    samples_per_step: int
    generator: Callable[[int], torch.Tensor]
    regeneration_frequency: Optional[int]


@dataclass
class Metrics(metaclass=ABCMeta):
    pass

    @abstractmethod
    def create_print_dict(self) -> Dict[str, float]:
        pass


_Metrics = TypeVar('_Metrics', bound='Metrics')


class Trainer(torch.nn.Module, Generic[_Metrics], metaclass=ABCMeta):

    def __init__(self, trainer_config: TrainerConfig):
        super().__init__()
        self.trainer_config = trainer_config
        self.counter = 0

    @abstractmethod
    def step(self, inputs: torch.Tensor) -> _Metrics:
        pass

    def batch_inputs(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        return [
            inputs[batch_start:batch_start + self.trainer_config.batch_size]
            for batch_start in range(0, inputs.shape[0], self.trainer_config.batch_size)
        ]

    def fit(self, number_of_steps: int, callbacks: List[Callable]) -> None:
        regeneration_frequency = self.trainer_config.regeneration_frequency
        for step_number in range(number_of_steps):
            if step_number % regeneration_frequency == 0 and regeneration_frequency is not None:
                inputs = self.trainer_config.generator(self.trainer_config.samples_per_step)
            self.counter += 1
            # noinspection PyUnboundLocalVariable
            metrics = self.step(inputs)
            self.run_callbacks(callbacks, metrics, self)

    @staticmethod
    def run_callbacks(callbacks: List[Callable], metrics: Metrics, trainer: 'Trainer') -> None:
        for callback in callbacks:
            callback(metrics, trainer)
