import functools
from dataclasses import dataclass
from typing import TypeVar, Union

import torch
from sigkernel import LinearKernel, RBFKernel, SigKernel

from src.config import DEVICE
from src.penalty.Metric import Metric, MetricConfig
from src.penalty.SigWassersteinMetric import SignatureConfig, SignatureEngine

_StaticKernel = TypeVar('_StaticKernel', bound=Union[LinearKernel, RBFKernel])


@dataclass
class SignatureMMDConfig(MetricConfig):
    signature_config: SignatureConfig
    static_kernel: _StaticKernel
    dyadic_order: int

    def __post_init__(self):
        self.sig_kernel = SigKernel(static_kernel=self.static_kernel, dyadic_order=self.dyadic_order)


class SignatureMMD(Metric[SignatureMMDConfig]):

    @property
    @functools.lru_cache()
    def signature_engine(self) -> SignatureEngine:
        return SignatureEngine(self.config.signature_config)

    @property
    @functools.lru_cache()
    def original_augmented(self) -> torch.Tensor:
        return self.signature_engine.augment_data(self.original).to(torch.double)

    def forward(self, generated: torch.Tensor):
        selected_original_augmented = self.sample_from_original_augmented(generated.shape[0])
        generated_augmented = self.signature_engine.augment_data(generated)
        return self.config.sig_kernel.compute_mmd(selected_original_augmented, generated_augmented)

    def sample_from_original_augmented(self, batch_size: int) -> torch.Tensor:
        return self.original_augmented[self.sample_indices(batch_size)].clone().to(DEVICE)



