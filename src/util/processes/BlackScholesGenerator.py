from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.util.processes.DiffusionGenerator import DiffusionGenerator


@dataclass
class BlackScholesParameterSet:
    drift: float
    sigma: float


class BlackScholesGenerator(DiffusionGenerator):

    def __init__(self, drift: NDArray, sigma: NDArray, covariance: NDArray = None):
        self.verify_dimension_of_drift_and_sigma(drift, sigma)
        self.drift = drift
        self.sigma = sigma
        super().__init__(DiffusionGenerator.get_covariance(self.drift.shape[0], covariance))

    def _generate(
            self,
            initial_value: NDArray,
            times: NDArray,
            stochastic_increments: Optional[NDArray] = None
    ) -> NDArray:
        time_increments = np.diff(times, 1)[np.newaxis, :, np.newaxis]
        drift, sigma = self.drift[np.newaxis, np.newaxis, :], self.sigma[np.newaxis, np.newaxis, :]
        return np.cumsum(
            np.concatenate([
                initial_value[:, np.newaxis, :],
                time_increments * drift + sigma * np.sqrt(time_increments) * stochastic_increments,
            ], axis=1),
            axis=1,
        )

    @staticmethod
    def verify_dimension_of_drift_and_sigma(drift: NDArray, sigma: NDArray) -> None:
        if drift.shape[0] != sigma.shape[0]:
            raise AttributeError(
                f'Drift and sigma must have coinciding dimensions, here {drift.shape[0]} and {sigma.shape[0]}'
            )


