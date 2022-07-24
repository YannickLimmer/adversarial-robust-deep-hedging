from _py_abc import ABCMeta
from abc import ABC
from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray

from src.util.processes.StochasticProcessGenerator import StochasticProcessGenerator


class DiffusionGenerator(StochasticProcessGenerator, ABC):

    def __init__(self, covariance: NDArray = None):
        self.covariance = covariance

    def shape_of_stochastic_increments(
            self,
            number_of_realizations: int,
            number_of_time_steps: int,
            number_of_dimensions: int,
    ) -> Tuple:
        return number_of_realizations, number_of_time_steps, number_of_dimensions

    def _generate_stochastic_increments(
            self,
            number_of_realizations: int,
            number_of_time_steps: int,
            number_of_dimensions: int,
            rng: np.random.Generator,
    ) -> NDArray:
        covariance = self.get_covariance(number_of_dimensions, self.covariance)
        return rng.multivariate_normal(
            np.zeros(number_of_dimensions),
            covariance,
            size=(number_of_realizations, number_of_time_steps),
        )

    @staticmethod
    def get_covariance(n_dim: int, covariance: Optional[NDArray]) -> NDArray:
        if covariance is None:
            return np.identity(n_dim)
        if covariance.shape[0] != n_dim or covariance.shape[1] != n_dim:
            raise AttributeError(
                f'Covariance shape {covariance.shape} does not agree with dimension of asset ({n_dim})'
            )
        return covariance
