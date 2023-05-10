import numpy as np
import torch

from src.config import DEVICE
from src.deep_hedging.AbstractDeepHedge import DeepHedgeConfig
from src.deep_hedging.DeepHedge import DeepHedge
from src.deep_hedging.StrategyNet import StrategyNetConfig
from src.deep_hedging.objectives.HedgeObjective import Entropy, StableEntropy, MeanVariance
from src.derivative.EuropeanCallOption import EuropeanCallOption
from src.penalty.CompareVolatility import CompareVolatility, VolatilityComparisonConfig, CompareVolatilityDynamically, \
    DynamicVolatilityComparisonConfig
from src.util.TimeUtil import UniformTimeDiscretization
from src.util.processes.BlackScholesGenerator import BlackScholesGenerator, BlackScholesParameterSet
from src.util.torch_util.AdapterUtil import ConvertToIncrements

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


def test_bs_process_vola():

    config = VolatilityComparisonConfig(td)
    p = CompareVolatility(bs_process, config)

    p(bs_process)


def test_dynamic_comparison():

    derivative = EuropeanCallOption(strike=1.0, time_discretization=td, price=0.0)
    dh = DeepHedge(
        DeepHedgeConfig(
            derivative=derivative,
            initial_information_value=torch.tensor([1.0], dtype=torch.float32, device=DEVICE),
            strategy_config=StrategyNetConfig(
                dim_of_information_process=1,
                dim_of_tradable_asset=1,
                number_of_layers=2,
                nodes_in_intermediate_layers=128,
            ),
        )
    )

    config = DynamicVolatilityComparisonConfig(td, hedge_objective=MeanVariance(130))
    p = CompareVolatilityDynamically(bs_process, dh, config)
    assert p(bs_process).item() == 0
    assert p(bs_generator(100)).item() > 0
