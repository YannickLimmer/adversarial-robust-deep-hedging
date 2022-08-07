import torch

from src.generator.HestonCoefficient import HestonDriftCoefficient, HestonDiffusionCoefficient
from src.generator.SdeGenerator import SdeGenerator


class HestonGenerator(SdeGenerator[HestonDriftCoefficient, HestonDiffusionCoefficient]):

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        process = super().forward(noise)
        ttm = torch.as_tensor(self.config.td.time_to_maturity, dtype=torch.float32)
        correction = (process[:, :, 1] - self.drift.reversion_level) / self.drift.reversion_speed \
            * (1 - torch.exp(- self.drift.reversion_speed * ttm)) + self.drift.reversion_level * ttm
        extended_time_increments = torch.cat((
            torch.tensor([0]),
            torch.as_tensor(self.config.td.time_step_increments, dtype=torch.float32),
        ))
        process[:, :, 1] = torch.cumsum(process[:, :, 1] * extended_time_increments, dim=1) + correction
        return process
