from typing import Dict

import numpy as np
import torch

from src.deep_hedging.AbstractDeepHedge import DeepHedgeConfig
from src.deep_hedging.DeepHedge import DeepHedge
from src.deep_hedging.DeepHedgeTrainer import DeepHedgeTrainer, DeepHedgeTrainerConfig
from src.deep_hedging.StrategyNet import StrategyNetConfig
from src.deep_hedging.objectives.HedgeObjective import MeanVariance
from src.derivative.EuropeanCallOption import EuropeanCallOption
from src.util.TimeUtil import UniformTimeDiscretization
from src.util.processes.BlackScholesGenerator import BlackScholesGenerator
from src.util.torch_util.TrainingUtil import TrainerConfig, PbarOption


def test_dh_trainer():
    torch.set_num_threads(10)

    trading_freq: int = 5
    td = UniformTimeDiscretization(trading_freq * 1./255., 90 // trading_freq)
    derivative = EuropeanCallOption(strike=1.0, time_discretization=td, price=0.0)
    initial_asset_price = np.array([1.0])
    strategy_config = StrategyNetConfig(
        dim_of_tradable_asset=1,
        dim_of_information_process=1,
        number_of_layers=3,
        nodes_in_intermediate_layers=36
    )
    config = DeepHedgeConfig(derivative, torch.Tensor(initial_asset_price), strategy_config)

    dh = DeepHedge(config)
    obj = MeanVariance(84)
    optimizer = torch.optim.Adam(dh.parameters())
    generator = BlackScholesGenerator(drift=np.array([0.05]), sigma=np.array([0.2])).provide_increment_generator(
        initial_value=np.array([1.0]),
        times=td.times,
        random_number_generator=np.random.default_rng(101)
    )

    def print_metrics(metrics: Dict[str, torch.Tensor], trainer: DeepHedgeTrainer) -> None:
        print(f"Iter: {trainer.counter}\t Loss: {metrics['hedge_loss']:.6f}")

    dh_trainer = DeepHedgeTrainer(
        DeepHedgeTrainerConfig(dh, obj, optimizer),
        TrainerConfig(100000, generator, 1)
    )

    dh_trainer.fit([1000]*5, pbar_option=PbarOption.EPOCH_BAR)

    dh.eval()
    inputs = generator(5000)
    res = np.sum(dh(inputs).detach().numpy() * inputs.numpy(), axis=(1, 2))
    term_a_v = np.sum(inputs.numpy(), axis=1)[:, 0] + initial_asset_price[0]

    # plt.scatter(term_a_v, res)
    # plt.show()

    assert obj(torch.as_tensor(res - np.maximum(term_a_v - derivative.strike, 0.0))).item() < 0.055
