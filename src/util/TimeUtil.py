from typing import List

import numpy as np


class TimeDiscretization:

    def __init__(self, time_steps: np.array):
        self.times = time_steps

    @property
    def time_steps(self) -> np.array:
        return self.times[1:]

    @property
    def time_step_increments(self) -> np.array:
        return np.diff(self.times, 1)

    @property
    def maturity(self) -> float:
        return self.times[-1]

    @property
    def number_of_time_steps(self) -> int:
        return self.times.shape[0] - 1

    @property
    def indices(self) -> List[int]:
        return list(range(self.number_of_time_steps))

    @property
    def time_to_maturity(self) -> np.array:
        return self.maturity - self.times


class UniformTimeDiscretization(TimeDiscretization):

    def __init__(self, time_step_size: float, number_of_time_steps):
        self.time_step_size = time_step_size
        super(UniformTimeDiscretization, self).__init__(
            np.linspace(0, time_step_size * number_of_time_steps, number_of_time_steps + 1, endpoint=True)
        )
