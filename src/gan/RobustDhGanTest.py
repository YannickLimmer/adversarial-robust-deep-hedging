from typing import Dict

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.deep_hedging.DeepHedge import DeepHedge
from src.deep_hedging.AbstractDeepHedge import DeepHedgeConfig
from src.deep_hedging.DeepHedgeTrainer import DeepHedgeTrainer, DeepHedgeTrainerConfig
from src.deep_hedging.StrategyNet import StrategyNetConfig
from src.deep_hedging.objectives.HedgeObjective import MeanVariance
from src.derivative.EuropeanCallOption import EuropeanCallOption
from src.gan.RobustDhGan import RobustDhGan
from src.generator.HestonCoefficient import HestonCoefficientConfig, HestonDriftCoefficient, HestonDiffusionCoefficient
from src.generator.SdeGenerator import GeneratorConfig, SdeGenerator
from src.generator.SdeGeneratorTrainer import SdeGeneratorTrainerConfig
from src.penalty.Augmentations import Scale, AddLags, LeadLag
from src.penalty.SigWassersteinMetric import SignatureConfig, SigWassersteinMetric
from src.util.TimeUtil import UniformTimeDiscretization
from src.util.processes.BlackScholesGenerator import BlackScholesGenerator
from src.util.processes.BrownianMotionGenerator import BrownianMotionGenerator
from src.util.processes.HestonGenerator import HestonParameterSet, HestonGenerator
from src.util.torch_util.AdapterUtil import Adapter, AdapterList, SelectDimensions, ConvertToIncrements
from src.util.torch_util.TrainingUtil import TrainerConfig
from src.util.torch_util.CallbackUtil import PrintMetrics, PrintGeneratorParameters, PrintEmptyLine

if __name__ == '__main__':
    # Initialize market data
    trading_freq: int = 5
    td = UniformTimeDiscretization(trading_freq * 1. / 255., 90 // trading_freq)
    derivative = EuropeanCallOption(strike=1.0, time_discretization=td, price=0.0)
    heston_parameters = HestonParameterSet(
        drift=0.05,
        reversion_level=0.04,
        reversion_speed=2,
        vol_of_vol=0.2,
        correlation=0.8,
    )
    heston_generator = HestonGenerator(parameter=heston_parameters).provide_generator(
        initial_value=np.array([1.0, 0.04]),
        times=td.times,
        random_number_generator=np.random.default_rng(404))
    heston_process = heston_generator(100000)

    # Initialize Penalizer
    sig_config = SignatureConfig(depth=2, augmentations=[Scale(0.2), AddLags(m=2), LeadLag()])
    penalizer = SigWassersteinMetric(heston_process[:, :, 0:1], sig_config, scale=1000)

    # Initialize Noise Generator
    noise_generator = BrownianMotionGenerator().provide_increment_generator(
        initial_value=np.zeros(2),
        times=td.times,
        random_number_generator=np.random.default_rng(101),
    )

    # Initialize Market Generator
    coef_config = HestonCoefficientConfig(heston_parameters, initial_asset_price=1.0)
    drift_coef, diffusion_coef = HestonDriftCoefficient(coef_config), HestonDiffusionCoefficient(coef_config)
    gen_config = GeneratorConfig(td, drift_coef.get_initial_asset_price, drift_coef, diffusion_coef)
    generator = SdeGenerator(generator_config=gen_config)
    gen_optimizer = torch.optim.Adam(generator.parameters())

    # Initialize Deep Hedge
    strategy_config = StrategyNetConfig(dimension_of_asset=1, number_of_layers=3, nodes_in_intermediate_layers=36)
    initial_asset_price_for_deep_hedge = torch.tensor([1.0])
    dh = DeepHedge(DeepHedgeConfig(derivative, initial_asset_price_for_deep_hedge, strategy_config))
    dh_optimizer = torch.optim.Adam(dh.parameters())
    hedge_objective = MeanVariance(84)

    # Initialize Trainer
    hedge_adapters = AdapterList([SelectDimensions(slice(0, 1)), ConvertToIncrements()])
    gen_adapters = AdapterList([SelectDimensions(slice(0, 1))])
    gen_train_config = SdeGeneratorTrainerConfig(generator, penalizer, gen_optimizer, gen_adapters)
    dh_train_config = DeepHedgeTrainerConfig(dh, hedge_objective, dh_optimizer, hedge_adapters)

    robust_dh_gan = RobustDhGan(
        hedge_config=dh_train_config,
        gen_config=gen_train_config,
        trainer_config=TrainerConfig(1000, 100000, noise_generator, 1),
    )
    robust_dh_gan.deactivate_generation_training()
    robust_dh_gan.fit(5, callbacks=[PrintMetrics()])

    robust_dh_gan.activate_generation_training()
    robust_dh_gan.fit(5, callbacks=[PrintMetrics(), PrintGeneratorParameters(), PrintEmptyLine()])
