import numpy as np
import torch

from src.deep_hedging.DeepHedge import DeepHedge
from src.deep_hedging.AbstractDeepHedge import DeepHedgeConfig
from src.deep_hedging.StrategyNet import StrategyNetConfig
from src.deep_hedging.objectives.HedgeObjective import MeanVariance
from src.derivative.EuropeanCallOption import EuropeanCallOption
from src.generator.HestonCoefficient import HestonDiffusionCoefficient, HestonCoefficientConfig, HestonDriftCoefficient
from src.generator.HestonGenerator import HestonGenerator
from src.generator.EulerGenerator import EulerGeneratorConfig
from src.util.TimeUtil import UniformTimeDiscretization
from src.util.processes.BrownianMotionGenerator import BrownianMotionGenerator
from src.util.processes.HestonGenerator import HestonParameterSet

heston_parameters = HestonParameterSet(
    drift=0.0,
    reversion_level=0.04,
    reversion_speed=1,
    vol_of_vol=0.2,
    correlation=0.8,
)

trading_freq: int = 2
td = UniformTimeDiscretization(trading_freq * 1. / 255., 90 // trading_freq)
derivative = EuropeanCallOption(strike=1.0, time_discretization=td, price=0.0)


def test_dh_heston():

    # TODO. This test does not work.

    coef_config = HestonCoefficientConfig(heston_parameters, initial_asset_price=1.0)
    drift_coef, diffusion_coef = HestonDriftCoefficient(coef_config), HestonDiffusionCoefficient(coef_config)
    gen_config = EulerGeneratorConfig(td, drift_coef.get_initial_asset_price, drift_coef, diffusion_coef)
    generator = HestonGenerator(generator_config=gen_config)
    generator.drift.drift.requires_grad = False

    noise_generator = BrownianMotionGenerator().provide_increment_generator(
        initial_value=np.zeros(2),
        times=td.times,
        random_number_generator=np.random.default_rng(4444),
    )

    strategy_config = StrategyNetConfig(
        dim_of_information_process=2,
        dim_of_tradable_asset=2,
        number_of_layers=3,
        nodes_in_intermediate_layers=36,
    )
    initial_asset_price_for_deep_hedge = torch.tensor([1.0, heston_parameters.reversion_level], dtype=torch.float32)
    deep_hedge = DeepHedge(DeepHedgeConfig(derivative, initial_asset_price_for_deep_hedge, strategy_config))

    sample_size = 100000
    batch_size = 1000
    hedge_objective = MeanVariance(84)

    generator.eval()
    generated = generator(noise_generator(sample_size)).detach()
    opt = torch.optim.Adam(deep_hedge.parameters())

    for batch in [generated[bno:min(bno + batch_size, sample_size)] for bno in range(0, sample_size, batch_size)]:
        pnl = deep_hedge(batch)
        loss = hedge_objective(pnl)
        deep_hedge.zero_grad()
        loss.backward()
        opt.step()
        # print(loss.item())