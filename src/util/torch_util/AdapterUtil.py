from _py_abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Sequence, List, Optional

import torch

Slice = TypeVar("Slice", bound=slice)


class Adapter(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        pass


class ConvertToIncrements(Adapter):

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.diff(inputs, n=1, dim=1)


@dataclass
class SelectDimensions(Adapter):

    dims: Slice

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :, self.dims]


@dataclass
class AdapterList(Adapter):

    adapters: Optional[List[Adapter]] = None

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._apply_adapter(inputs, self.adapters.copy())

    def _apply_adapter(self, inputs: torch.Tensor, adapters: List[Adapter]) -> torch.Tensor:
        if adapters:
            adapter = adapters.pop()
            return adapter(self._apply_adapter(inputs, adapters))
        else:
            return inputs