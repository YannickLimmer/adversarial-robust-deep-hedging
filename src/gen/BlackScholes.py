from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy._typing import NDArray

from src.gen.Generator import EulerGenerator, DiffusionGeneratorConfig
from src.util.TimeUtil import TimeDiscretization, UniformTimeDiscretization
from src.util.torch_util.ParameterUtil import ExpParameter, TanhParameter, TransformedParameter


@dataclass
class BlackScholesParameterSet:
    sigma: Union[float, NDArray]
    drift: Union[float, NDArray] = 0.0

    def __post_init__(self):
        if isinstance(self.sigma, float):
            self.sigma = np.array([[self.sigma]])
            self.dim = 1
        else:
            self.dim = self.sigma.shape[0]
        if isinstance(self.drift, float):
            self.drift = np.ones(self.dim) * self.drift


class BlackScholesGeneratorConfig(DiffusionGeneratorConfig):

    def __init__(self, td: TimeDiscretization, default_pars: BlackScholesParameterSet):
        super().__init__(td, process_dim=default_pars.dim, noise_dim=default_pars.dim)
        self.default_pars = default_pars


class BlackScholesGenerator(EulerGenerator[BlackScholesGeneratorConfig]):

    def _register__(self):
        self.sigma = TransformedParameter(
            torch.tensor(self.config.default_pars.sigma, dtype=self.dtype, device=self.device)
        )
        self.drift_par = TransformedParameter(
            torch.tensor(self.config.default_pars.drift, dtype=self.dtype, device=self.device),
            requires_grad=False,
        )

    def drift(self, t: float, prev: torch.Tensor) -> torch.Tensor:
        return prev * self.drift_par.val

    def diffusion(self, t: float, prev: torch.Tensor) -> torch.Tensor:
        return prev[:, :, None] * self.sigma.val[None, :, :]

    @property
    def default_initial(self) -> torch.Tensor:
        return torch.ones(self.config.process_dim, dtype=self.dtype, device=self.device)


if __name__ == '__main__':
    s = np.eye(3) * 0.3
    s[1, 0] = .1
    pars = BlackScholesParameterSet(s, drift=1)
    tdis = UniformTimeDiscretization.from_bounds(0, 1, 60)
    conf = BlackScholesGeneratorConfig(tdis, pars)
    gen = BlackScholesGenerator(conf)
    p = gen(100).detach()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 9))
    ax1.plot(tdis.times, p.numpy()[:, :, 0].T)
    ax2.plot(tdis.times, p.numpy()[:, :, 1].T)
    ax3.plot(tdis.times, p.numpy()[:, :, 2].T)
    plt.show()
