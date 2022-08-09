from typing import List

import numpy as np
from numpy.typing import NDArray


class QuantityLogger:

    def __init__(self):
        self.history: List[float] = []
        self.mark = 0

    def set_mark(self):
        self.mark = len(self.history)

    @property
    def average_since_mark(self) -> NDArray:
        return np.mean(self.history[self.mark:])

    @property
    def average(self) -> NDArray:
        return np.mean(self.history)

    def update(self, val: float) -> None:
        self.history.append(val)
