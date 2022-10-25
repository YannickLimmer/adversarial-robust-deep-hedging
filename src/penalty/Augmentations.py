from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from src.config import DEVICE
from src.util.TimeUtil import TimeDiscretization


@dataclass
class BaseAugmentation(metaclass=ABCMeta):
    pass

    @abstractmethod
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        pass


@dataclass
class AddTimeComponent(BaseAugmentation):

    td: TimeDiscretization

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        return torch.cat((get_time_steps(data.shape[0], self.td), data), dim=2)


@dataclass
class Scale(BaseAugmentation):
    scale: float = 1

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        return self.scale * data


@dataclass
class Cumsum(BaseAugmentation):
    dim: int = 1

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        return data.cumsum(dim=self.dim)


@dataclass
class AddLags(BaseAugmentation):
    m: int = 2

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        return self.cat_lags(data, self.m)

    @staticmethod
    def cat_lags(data: torch.Tensor, number_of_lags: int) -> torch.Tensor:
        AddLags.verify_number_of_lags_is_applicable(number_of_lags, data.shape[1])
        return torch.cat([data.clone()[:, i:i + number_of_lags] for i in range(number_of_lags)], dim=-1)

    @staticmethod
    def verify_number_of_lags_is_applicable(number_of_lags: int, number_of_time_steps: int) -> None:
        if number_of_time_steps < number_of_lags:
            raise AttributeError('Lift cannot be performed. q < m : (%s < %s)' % (number_of_time_steps, number_of_lags))


@dataclass
class LeadLag(BaseAugmentation):
    """
    Lead-lag transformation for a multivariate paths.
    """

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        return self.lead_lag_transform(data)

    @staticmethod
    def lead_lag_transform(data: torch.Tensor) -> torch.Tensor:
        data_repeated = torch.repeat_interleave(data, repeats=2, dim=1)
        return torch.cat([data_repeated[:, :-1], data_repeated[:, 1:]], dim=2)


@dataclass
class LeadLagWithTime(BaseAugmentation):
    """
    Lead-lag transformation for a multivariate paths with time component.
    """
    td: TimeDiscretization

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        return self.lead_lag_transform_with_time(data, self.td)

    @staticmethod
    def lead_lag_transform_with_time(data: torch.Tensor, td: TimeDiscretization) -> torch.Tensor:
        time_steps = get_time_steps(data.shape[0], td).to(DEVICE)
        time_steps_repeated = torch.repeat_interleave(time_steps, repeats=3, dim=1)
        data_repeated = torch.repeat_interleave(data, repeats=3, dim=1)
        return torch.cat([time_steps_repeated[:, 0:-2], data_repeated[:, 1:-1], data_repeated[:, 2:]], dim=2)


def get_time_steps(batch_size: int, td: TimeDiscretization) -> torch.Tensor:
    return torch.from_numpy(np.copy(td.times))[None, :, None].repeat(batch_size, 1, 1)
