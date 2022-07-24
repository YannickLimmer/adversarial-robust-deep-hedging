from dataclasses import dataclass
from functools import cached_property
from typing import Tuple, List

import numpy as np
import signatory
import torch

from src.config import DEVICE
from src.penalty.Augmentations import BaseAugmentation


@dataclass
class SignatureConfig:
    depth: int
    basepoint: bool = False
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


class SigWassersteinMetric(torch.nn.Module):

    def __init__(self, original: torch.Tensor, signature_config: SignatureConfig, scale: float=1.0):
        super().__init__()
        self.original = original
        self.signature_engine = SignatureEngine(signature_config)
        self.scale = scale

    @cached_property
    def original_signatures(self) -> torch.Tensor:
        return self.signature_engine.compute_signatures(self.original)

    def forward(self, generated: torch.Tensor) -> torch.Tensor:
        selected_original_signatures = self.sample_from_original_signatures(generated.shape[0])
        generated_signatures = self.signature_engine.compute_signatures(generated)
        return self.compute_loss(selected_original_signatures, generated_signatures) * self.scale

    def sample_from_original_signatures(self, batch_size: int) -> torch.Tensor:
        return self.original_signatures[self.sample_indices(batch_size)].clone().to(DEVICE)

    def sample_indices(self, batch_size: int):
        return torch.from_numpy(
            np.random.choice(self.original.shape[0], size=batch_size, replace=False),
        ).to(DEVICE)

    @staticmethod
    def compute_loss(signatures_original: torch.Tensor, signatures_generated: torch.Tensor) -> torch.Tensor:
        return torch.norm(torch.mean(signatures_original, dim=0) - torch.mean(signatures_generated, dim=0), p=2, dim=0)
