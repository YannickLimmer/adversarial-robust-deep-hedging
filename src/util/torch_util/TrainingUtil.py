from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from enum import Enum
from typing import List, Callable, Optional, Dict, TypeVar, Generic
from tqdm.auto import tqdm

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

    @classmethod
    @abstractmethod
    def summarize_metrics_list_to_metrics(cls, metrics_list: List['Metrics']) -> 'Metrics':
        pass


_Metrics = TypeVar('_Metrics', bound='Metrics')


class PbarOption(Enum):
    BATCH_BAR = 'A progress bar over all batches that remains and a progress bar over the epochs.'
    VANISHING_BATCH_BAR = 'A progress bar over all batches that vanishes and a progress bar over the epochs.'
    EPOCH_BAR = 'Only a progress bar over the epochs'
    NO_BAR = 'No progress bar'

    @property
    def has_epoch_bar(self) -> bool:
        return self is not PbarOption.NO_BAR

    @property
    def has_batch_bar(self) -> bool:
        return self not in {PbarOption.NO_BAR, PbarOption.EPOCH_BAR}

    @property
    def has_remaining_batch_bar(self) -> bool:
        return self is PbarOption.BATCH_BAR


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

    def fit(self, number_of_steps: int, pbar_option: PbarOption = PbarOption.VANISHING_BATCH_BAR) -> None:
        inputs = None

        pbar = tqdm(range(number_of_steps), leave=True) if pbar_option.has_epoch_bar else range(number_of_steps)
        for step_number in pbar:
            inputs = self.generate_inputs(inputs, step_number)
            metrics = self.fit_on_batches(inputs, pbar_option)
            pbar.set_postfix(metrics.create_print_dict())
            self.counter += 1

    def fit_on_batches(self, inputs: torch.Tensor, pbar_option: PbarOption) -> Metrics:
        batches = self.batch_inputs(inputs)

        m_list = []
        pbar = tqdm(batches, leave=pbar_option.has_remaining_batch_bar) if pbar_option.has_epoch_bar else batches
        for batch in pbar:
            m = self.step(batch)
            m_list.append(m)
            pbar.set_postfix(m.create_print_dict())

        return _Metrics.summarize_metrics_list_to_metrics(m_list)

    def generate_inputs(self, inputs: Optional[torch.Tensor], step_number: int) -> torch.Tensor:
        reg_fr = self.trainer_config.regeneration_frequency
        if (reg_fr is None and step_number > 0) or (reg_fr is not None and step_number % reg_fr > 0):
            return inputs
        return self.trainer_config.generator(self.trainer_config.samples_per_step)

    @staticmethod
    def run_callbacks(callbacks: List[Callable], metrics: List[Metrics], trainer: 'Trainer') -> None:
        for callback in callbacks:
            callback(metrics, trainer)
