import functools
import math
from dataclasses import dataclass
from typing import List

import signatory
import torch

from src.config import DEVICE
from src.penalty.Augmentations import BaseAugmentation
from src.penalty.Metric import Metric, MetricConfig


@dataclass
class SignatureConfig(MetricConfig):
    depth: int
    basepoint: bool = False
    normalize: bool = True
    augmentations: List[BaseAugmentation] = ()


class SignatureEngine:

    def __init__(self, signature_config: SignatureConfig):
        self.config = signature_config

    def compute_signatures(self, data: torch.Tensor) -> torch.Tensor:
        augmented_data = self.augment_data(data)
        return signatory.signature(augmented_data, self.config.depth, basepoint=self.config.basepoint)

    def augment_data(self, data: torch.Tensor) -> torch.Tensor:
        for augmentation in self.config.augmentations:
            data = augmentation.apply(data)
        return data


class SigWassersteinMetric(Metric[SignatureConfig]):

    @property
    @functools.lru_cache()
    def signature_engine(self) -> SignatureEngine:
        return SignatureEngine(self.config)

    @property
    @functools.lru_cache()
    def original_signatures(self) -> torch.Tensor:
        return self.signature_engine.compute_signatures(self.original)

    def forward(self, generated: torch.Tensor) -> torch.Tensor:
        selected_original_signatures = self.sample_from_original_signatures(generated.shape[0])
        generated_signatures = self.signature_engine.compute_signatures(generated)
        return self.transform(self.compute_loss(selected_original_signatures, generated_signatures))

    def sample_from_original_signatures(self, batch_size: int) -> torch.Tensor:
        return self.original_signatures[self.sample_indices(batch_size)].clone().to(DEVICE)

    def compute_loss(self, signatures_original: torch.Tensor, signatures_generated: torch.Tensor) -> torch.Tensor:
        return torch.mean(
            (self.compute_esig(signatures_original, dim=0) - self.compute_esig(signatures_generated, dim=0)) ** 2,
        )

    def compute_esig(self, signatures: torch.Tensor, dim: int) -> torch.Tensor:
        esig = torch.mean(signatures, dim=dim)
        if self.signature_engine.config.normalize:
            count = 0
            for i in range(self.signature_engine.config.depth):
                esig[count:count + dim ** (i + 1)] = esig[count:count + dim ** (i + 1)] * math.factorial(i + 1)
                count = count + dim ** (i + 1)
        return esig

