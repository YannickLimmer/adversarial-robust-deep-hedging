from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.util.processes.DiffusionGenerator import DiffusionGenerator


class BrownianMotionGenerator(DiffusionGenerator):

    def _generate(
            self,
            initial_value: NDArray,
            times: NDArray,
            stochastic_increments: Optional[NDArray] = None,
    ) -> NDArray:
        time_increments = np.diff(times, 1)[np.newaxis, :, np.newaxis]
        return np.cumsum(
            np.concatenate([initial_value[:, np.newaxis, :], np.sqrt(time_increments) * stochastic_increments], axis=1),
            axis=1,
        )


