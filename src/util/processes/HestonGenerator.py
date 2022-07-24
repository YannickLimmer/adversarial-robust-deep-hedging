from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy._typing import NDArray

from src.util.processes.DiffusionGenerator import DiffusionGenerator
from src.util.processes.EulerMaruyamaGenerator import EulerMaruyamaGenerator


@dataclass
class HestonParameterSet:
    drift: float
    reversion_speed: float
    reversion_level: float
    vol_of_vol: float
    correlation: float


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
        target_matrix = np.array([[1.0, self.pars.correlation], [self.pars.correlation, 1.0]])
        return np.linalg.cholesky(target_matrix)

