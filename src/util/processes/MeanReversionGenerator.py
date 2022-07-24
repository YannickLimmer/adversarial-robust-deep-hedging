from abc import ABC
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.util.processes.BrownianMotionGenerator import BrownianMotionGenerator
from src.util.processes.DiffusionGenerator import DiffusionGenerator


class MeanReversionGenerator(DiffusionGenerator, ABC):

    def __init__(self, level: NDArray, sigma: NDArray, speed: NDArray, covariance: NDArray = None):
        self.verify_dimension_of_parameters(level, speed, sigma)
        self.level = level
        self.speed = speed
        self.sigma = sigma
        super().__init__(DiffusionGenerator.get_covariance(self.level.shape[0], covariance))

    @staticmethod
    def verify_dimension_of_parameters(level: NDArray, speed: NDArray, sigma: NDArray) -> None:
        if level.shape[0] != speed.shape[0]:
            raise AttributeError(
                f'Level and speed must have coinciding dimensions, here {level.shape[0]} and {speed.shape[0]}'
            )
        if level.shape[0] != sigma.shape[0]:
            raise AttributeError(
                f'Level and sigma must have coinciding dimensions, here {level.shape[0]} and {sigma.shape[0]}'
            )


class AnalyticMeanReversionGenerator(MeanReversionGenerator):

    def _generate(
            self,
            initial_value: NDArray,
            times: NDArray,
            stochastic_increments: Optional[NDArray] = None,
    ) -> NDArray:
        # TODO. There is some error in this computation / formula.
        bm = BrownianMotionGenerator(self.covariance).generate(
            initial_value=np.zeros_like(initial_value),
            times=times,
            stochastic_increments=stochastic_increments,
        )
        return self._calculate_process(
            brownian_motion=bm,
            initial_value=initial_value[:, np.newaxis, :],
            level=self.level[np.newaxis, np.newaxis, :],
            speed=self.speed[np.newaxis, np.newaxis, :],
            sigma=self.sigma[np.newaxis, np.newaxis, :],
            time_steps=times[1:],
        )

    @staticmethod
    def _calculate_process(
            brownian_motion: NDArray,
            initial_value: NDArray,
            level: NDArray,
            speed: NDArray,
            sigma: NDArray,
            time_steps: NDArray,
    ) -> NDArray:
        exponential = np.exp(- speed * time_steps)
        deterministic = initial_value * exponential + level * (1 - exponential)
        factor = (sigma * exponential / np.sqrt(2 * level))
        stochastic = factor * np.sqrt(np.exp(2 * speed * time_steps) - 1) * brownian_motion
        return deterministic + stochastic


class EulerMaruyamaMeanReversionGenerator(MeanReversionGenerator):

    def _generate(
            self,
            initial_value: NDArray,
            times: NDArray,
            stochastic_increments: Optional[NDArray] = None
            ) -> NDArray:
        values_at_times = [initial_value]
        for index, time_increment in enumerate(np.diff(times, 1)):
            drift_term = self.speed[np.newaxis, :] * (self.level[np.newaxis, :] - values_at_times[-1]) * time_increment
            diffusion_term = self.sigma[np.newaxis, :] * np.sqrt(time_increment) * stochastic_increments[:, index, :]
            values_at_times.append(values_at_times[-1] + drift_term + diffusion_term)
        return np.stack(values_at_times, axis=1)

