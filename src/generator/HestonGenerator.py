import torch

from src.config import DEVICE
from src.generator.HestonCoefficient import HestonDriftCoefficient, HestonDiffusionCoefficient
from src.generator.EulerGenerator import EulerGenerator


class HestonGenerator(EulerGenerator[HestonDriftCoefficient, HestonDiffusionCoefficient]):

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        process = super().forward(noise)
        correction = self.calculate_correction(process)
        volatility_swap = torch.cumsum(process[:, :, 1] * self.extended_time_increments, dim=1) + correction
        return torch.cat((process, volatility_swap[:, :, None]), dim=2)

    @property
    def extended_time_increments(self) -> torch.Tensor:
        return torch.cat((
            torch.tensor([0], device=DEVICE),
            torch.as_tensor(self.config.td.time_step_increments, dtype=torch.float32, device=DEVICE),
        ))

    def calculate_correction(self, process: torch.Tensor) -> torch.Tensor:
        ttm = torch.as_tensor(self.config.td.time_to_maturity, dtype=torch.float32, device=DEVICE)
        return (process[:, :, 1] - self.drift.reversion_level) / self.drift.reversion_speed \
            * (1 - torch.exp(- self.drift.reversion_speed * ttm)) + self.drift.reversion_level * ttm
