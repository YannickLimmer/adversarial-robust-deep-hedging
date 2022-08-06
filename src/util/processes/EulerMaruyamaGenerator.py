from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.util.processes.DiffusionGenerator import DiffusionGenerator


class EulerMaruyamaGenerator(DiffusionGenerator, metaclass=ABCMeta):

    def _generate(
            self,
            initial_value: NDArray,
            times: NDArray,
            stochastic_increments: Optional[NDArray] = None
            ) -> NDArray:
        time_increments = np.diff(times, 1)
        process = [initial_value]
        for index in range(time_increments.shape[0]):
            drift_increment = self.drift(process[-1], times[index]) * time_increments[index]
            noise = stochastic_increments[:, index, :, None] * np.sqrt(time_increments[index])
            stoch_increment = self.diffusion(process[-1], times[index]) @ noise
            process.append(process[-1] + drift_increment + stoch_increment[:, :, 0])

        return np.stack(process, axis=1)

    @abstractmethod
    def drift(self, process_at_time_before: NDArray, time: np.float) -> NDArray:
        pass

    @abstractmethod
    def diffusion(self, process_at_time_before: NDArray, time: np.float) -> NDArray:
        pass

