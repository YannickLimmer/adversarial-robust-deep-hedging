import numpy as np

from src.derivative.EuropeanCallOption import EuropeanCallOption
from src.penalty.CompareVolatility import CompareVolatility, VolatilityComparisonConfig
from src.util.TimeUtil import UniformTimeDiscretization
from src.util.processes.BlackScholesGenerator import BlackScholesGenerator, BlackScholesParameterSet


def test_bs_process_vola():

    trading_freq: int = 5
    td = UniformTimeDiscretization(trading_freq * 1. / 255., 90 // trading_freq)
    bs_parameters = BlackScholesParameterSet(
        drift=0.0,
        sigma=0.2,
    )
    bs_generator = BlackScholesGenerator(
        drift=np.array([bs_parameters.drift]), sigma=np.array([bs_parameters.sigma]),
    ).provide_generator(
        initial_value=np.array([1.0]),
        times=td.times,
        random_number_generator=np.random.default_rng(404)
    )
    bs_process = bs_generator(100000)

    config = VolatilityComparisonConfig(td)
    p = CompareVolatility(bs_process, config)

    p(bs_process)
