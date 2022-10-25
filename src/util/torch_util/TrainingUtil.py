from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from enum import Enum
from typing import List, Callable, Optional, Dict, TypeVar, Generic, get_args, Any

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

import torch

from src.util.torch_util.AdapterUtil import ConvertToIncrements


@dataclass
class TrainerConfig:
    samples_per_step: int
    generator: Callable[[int], torch.Tensor]
    regeneration_frequency: Optional[int]


@dataclass
class Metrics(metaclass=ABCMeta):
    pass

    @abstractmethod
    def create_print_dict(self) -> Dict[str, float]:
        pass

    @property
    @abstractmethod
    def loss(self) -> NDArray:
        pass

    @classmethod
    @abstractmethod
    def summarize_metrics_list_to_metrics(cls, metrics_list: List['Metrics']) -> 'Metrics':
        pass


_Metrics = TypeVar('_Metrics', bound='Metrics')


class PbarOption(Enum):
    BATCH_BAR = 'A progress bar over all batches that remains and a progress bar over the epochs.'
    VANISHING_BATCH_BAR = 'A progress bar over all batches that vanishes and a progress bar over the epochs.'
    VANISHING_BARS = 'A progress bar over all batches and a progress bar over the epochs that both vanish.'
    EPOCH_BAR = 'Only a progress bar over the epochs'
    VANISHING_EPOCH_BAR = 'Only a progress bar over the epochs that vanishes.'
    NO_BAR = 'No progress bar'

    @property
    def has_epoch_bar(self) -> bool:
        return self is not PbarOption.NO_BAR

    @property
    def has_batch_bar(self) -> bool:
        return self in {PbarOption.BATCH_BAR, PbarOption.VANISHING_BATCH_BAR, PbarOption.VANISHING_BARS}

    @property
    def has_remaining_batch_bar(self) -> bool:
        return self is PbarOption.BATCH_BAR

    @property
    def has_remaining_epoch_bar(self) -> bool:
        return self in {PbarOption.BATCH_BAR, PbarOption.EPOCH_BAR, PbarOption.VANISHING_BATCH_BAR}


class Trainer(torch.nn.Module, Generic[_Metrics], metaclass=ABCMeta):

    def __init__(self, trainer_config: TrainerConfig):
        super().__init__()
        self.trainer_config = trainer_config
        self.counter = 0
        self.metric_log = []

    @abstractmethod
    def step(self, inputs: torch.Tensor) -> _Metrics:
        pass

    @staticmethod
    def batch_inputs(inputs: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
        return [inputs[batch_start:batch_start + batch_size] for batch_start in range(0, inputs.shape[0], batch_size)]

    def load_or_fit(
            self,
            f: Any,
            batch_sizes: List[int],
            pbar_option: PbarOption = PbarOption.VANISHING_BATCH_BAR,
            pretrained: Optional[Any] = None,
            loss_curve_address: Optional[Any] = None,
            parameter_tracking_address: Optional[Any] = None,
    ) -> None:
        try:
            self.load_module_from_state_dict(f)
        except FileNotFoundError:
            self.fit(batch_sizes, pbar_option, pretrained, loss_curve_address, parameter_tracking_address)

    @abstractmethod
    def load_module_from_state_dict(self, f: Any) -> None:
        pass

    def fit(
            self,
            batch_sizes: List[int],
            pbar_option: PbarOption = PbarOption.VANISHING_BATCH_BAR,
            pretrained: Optional[Any] = None,
            loss_curve_address: Optional[Any] = None,
            parameter_tracking_address: Optional[Any] = None,
    ) -> None:
        if pretrained:
            self.load_module_from_state_dict(pretrained)

        bs_pbar = tqdm(
            list(enumerate(batch_sizes)),
            leave=pbar_option.has_remaining_epoch_bar,
        ) if pbar_option.has_epoch_bar else enumerate(batch_sizes)

        inputs = None
        losses = []
        parameters_for_epoch = []
        for epoch, batch_size in bs_pbar:
            inputs = self.generate_inputs(inputs, epoch)
            metrics = self.fit_on_batches(inputs, batch_size, pbar_option)
            bs_pbar.set_postfix(metrics.create_print_dict()) if pbar_option.has_epoch_bar else None
            losses.append(metrics.loss)
            parameters_for_epoch.append({k: v.clone().detach().numpy() for k, v in self.named_parameters()})
            self.counter += 1

        if loss_curve_address:
            np.save(loss_curve_address, np.stack(losses, axis=0))

        if parameter_tracking_address:
            np.save(parameter_tracking_address, parameters_for_epoch)

    def fit_on_batches(self, inputs: torch.Tensor, batch_size: int, pbar_option: PbarOption) -> Metrics:
        batches = self.batch_inputs(inputs, batch_size)

        m_list = []
        b_pbar = tqdm(batches, leave=pbar_option.has_remaining_batch_bar) if pbar_option.has_batch_bar else batches
        for batch in b_pbar:
            m = self.step(batch)
            m_list.append(m)
            b_pbar.set_postfix(m.create_print_dict()) if pbar_option.has_batch_bar else None

        self.metric_log.append(m_list)
        return get_args(type(self).__orig_bases__[0])[0].summarize_metrics_list_to_metrics(m_list)

    def generate_inputs(self, inputs: Optional[torch.Tensor], epoch: int) -> torch.Tensor:
        reg_fr = self.trainer_config.regeneration_frequency
        if (reg_fr is None and epoch > 0) or (reg_fr is not None and epoch % reg_fr > 0):
            return inputs
        return self.trainer_config.generator(self.trainer_config.samples_per_step)

    @staticmethod
    def run_callbacks(callbacks: List[Callable], metrics: List[Metrics], trainer: 'Trainer') -> None:
        for callback in callbacks:
            callback(metrics, trainer)


def gen_factory(path_gen: torch.nn.Module, noise_gen: Callable[[int], torch.Tensor]) -> Callable[[int], torch.Tensor]:
    to_increments = ConvertToIncrements()

    def gen(n: int) -> torch.Tensor:
        return to_increments(path_gen(noise_gen(n)).detach())

    return gen
