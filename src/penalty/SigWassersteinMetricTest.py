import numpy as np
import torch

from src.penalty.Augmentations import LeadLag, Scale, Cumsum, AddLags
from src.penalty.SigWassersteinMetric import SigWassersteinMetric, SignatureConfig
from src.util.TimeUtil import UniformTimeDiscretization
from src.util.processes.BlackScholesGenerator import BlackScholesGenerator

trading_freq: int = 10
td = UniformTimeDiscretization(trading_freq * 1. / 255., 90 // trading_freq)

bs_process = torch.as_tensor(
    BlackScholesGenerator(drift=np.array([0.0]), sigma=np.array([0.2])).generate(
        initial_value=np.ones((100000, 1)),
        times=td.times,
        random_number_generator=np.random.default_rng(101),
    ), dtype=torch.float32
)
# noinspection PyTypeChecker
sig_config = SignatureConfig(depth=3, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()]))
metric = SigWassersteinMetric(bs_process, sig_config)


def test_sig_wasserstein_identity():

    assert metric(bs_process).item() < 0.0001


def test_sig_wasserstein_on_drift_deviations():

    assert (np.diff([
        metric(
                torch.as_tensor(
                    BlackScholesGenerator(drift=np.array([drift]), sigma=np.array([0.2])).generate(
                        initial_value=np.ones((10000, 1)),
                        times=td.times,
                        random_number_generator=np.random.default_rng(104),
                    ), dtype=torch.float32
                )
        ).detach().item() for drift in np.linspace(0.0, 0.5, 6)
    ], 1) > 0).all()

    assert (np.diff([
        metric(
            torch.as_tensor(
                BlackScholesGenerator(drift=np.array([drift]), sigma=np.array([0.2])).generate(
                    initial_value=np.ones((100000, 1)),
                    times=td.times,
                    random_number_generator=np.random.default_rng(104),
                ), dtype=torch.float32
            )
        ).detach().item() for drift in np.linspace(0.0, -0.5, 6)
    ], 1) > 0).all()


def test_sig_wasserstein_on_sigma_deviations():

    assert (np.diff([
        metric(
            torch.as_tensor(
                BlackScholesGenerator(drift=np.array([0.0]), sigma=np.array([sigma])).generate(
                    initial_value=np.ones((100000, 1)),
                    times=td.times,
                    random_number_generator=np.random.default_rng(104),
                ), dtype=torch.float32
            )
        ).detach().item() for sigma in np.linspace(0.2, 0.7, 6)
    ], 1) > 0).all()

    assert (np.diff([
        metric(
            torch.as_tensor(
                BlackScholesGenerator(drift=np.array([0.0]), sigma=np.array([sigma])).generate(
                    initial_value=np.ones((100000, 1)),
                    times=td.times,
                    random_number_generator=np.random.default_rng(104),
                ), dtype=torch.float32
            )
        ).detach().item() for sigma in np.linspace(0.2, 0.0, 3)
    ], 1) > 0).all()
