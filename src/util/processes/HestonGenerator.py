from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.util.processes.EulerMaruyamaGenerator import EulerMaruyamaGenerator


@dataclass
class HestonParameterSet:
    drift: float
    reversion_speed: float
    reversion_level: float
    vol_of_vol: float
    correlation: float

    def normalize_drift(self, drift):
        return drift / self.drift if self.drift != 0 else self.drift

    def denormalize_drift(self, normalized_drift):
        return normalized_drift * self.drift if self.drift != 0 else self.drift

    def normalize_reversion_speed(self, reversion_speed):
        return reversion_speed / self.reversion_speed

    def denormalize_reversion_speed(self, normalized_reversion_speed):
        return normalized_reversion_speed * self.reversion_speed

    def normalize_reversion_level(self, reversion_level):
        return reversion_level / self.reversion_level

    def denormalize_reversion_level(self, normalized_reversion_level):
        return normalized_reversion_level * self.reversion_level

    def normalize_vol_of_vol(self, vol_of_vol):
        return vol_of_vol / self.vol_of_vol

    def denormalize_vol_of_vol(self, normalized_vol_of_vol):
        return normalized_vol_of_vol * self.vol_of_vol


class HestonGenerator(EulerMaruyamaGenerator):

    def __init__(self, parameter: HestonParameterSet):
        super().__init__()
        self.pars = parameter

    def drift(self, process_at_time_before: NDArray, time: np.float) -> NDArray:
        asset_component = process_at_time_before[:, 0] * self.pars.drift
        volatility_component = self.pars.reversion_speed * (self.pars.reversion_level - process_at_time_before[:, 1])
        return np.stack([asset_component, volatility_component], axis=1)

    def diffusion(self, process_at_time_before: NDArray, time: np.float) -> NDArray:
        root_of_vola = np.sqrt(np.abs(process_at_time_before[:, 1]))
        asset_component = process_at_time_before[:, 0] * root_of_vola
        volatility_component = self.pars.vol_of_vol * root_of_vola
        return np.stack([asset_component, volatility_component], axis=1)[:, :, None] \
               * self.correlation_matrix_root[None, :, :]

    @property
    def correlation_matrix_root(self) -> NDArray:
        return np.array([[1.0, 0], [self.pars.correlation, np.sqrt(1 - self.pars.correlation ** 2)]])

    def _generate(
            self,
            initial_value: NDArray,
            times: NDArray,
            stochastic_increments: Optional[NDArray] = None,
    ) -> NDArray:
        process = super()._generate(initial_value, times, stochastic_increments)
        ttm = (times[-1] - times)[None, :]
        volatility_process = process[:, :, 1]
        correction = self.get_correction_term(ttm, volatility_process)
        extended_time_increments = np.concatenate((np.array([0]), np.diff(times, 1)))[None, :]
        volatility_swap = np.cumsum(process[:, :, 1] * extended_time_increments, axis=1) + correction
        return np.concatenate((process, volatility_swap[:, :, np.newaxis]), axis=2)

    def get_correction_term(self, ttm: NDArray, volatility_process: NDArray) -> NDArray:
        return (volatility_process - self.pars.reversion_level) / self.pars.reversion_speed \
               * (1 - np.exp(- self.pars.reversion_speed * ttm)) + self.pars.reversion_level * ttm
