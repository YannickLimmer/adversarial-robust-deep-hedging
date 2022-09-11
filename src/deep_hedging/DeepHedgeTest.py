import numpy as np
import torch
from tqdm.auto import tqdm

from src.deep_hedging.DeepHedge import DeepHedge
from src.deep_hedging.AbstractDeepHedge import DeepHedgeConfig
from src.deep_hedging.StrategyNet import StrategyNetConfig
from src.derivative.EuropeanCallOption import EuropeanCallOption
from src.util.TimeUtil import UniformTimeDiscretization
from src.util.processes.BlackScholesGenerator import BlackScholesGenerator


def test_dh():
    torch.set_num_threads(10)

    print('Initialization...')
    trading_freq: int = 5
    td = UniformTimeDiscretization(trading_freq * 1./255., 90 // trading_freq)
    derivative = EuropeanCallOption(strike=1.0, time_discretization=td, price=0.0)
    initial_asset_price = np.array([1.0])
    strategy_config = StrategyNetConfig(
        dimension_of_asset=1,
        number_of_layers=3,
        nodes_in_intermediate_layers=36,
    )
    config = DeepHedgeConfig(derivative, torch.Tensor(initial_asset_price), strategy_config)
    dh = DeepHedge(config)
    print('--> Number_of_parameters:    ', sum(p.numel() for p in dh.parameters()))

    optimizer = torch.optim.Adam(dh.parameters())
    print('--> Done.')

    print('Data generation...')
    number_of_realizations = 10000
    inputs = torch.as_tensor(
        np.diff(BlackScholesGenerator(drift=np.array([0.0]), sigma=np.array([0.2])).generate(
            initial_asset_price.T * np.ones((number_of_realizations, 1)),
            td.times,
            random_number_generator=np.random.default_rng(101),
        ), 1, 1), dtype=torch.float32,
    )
    print('--> Done.')

    batch_size = 100
    for i in range(5):
        for batch in tqdm(range(0, number_of_realizations, batch_size), ):
            output = dh(inputs[batch:batch+batch_size])
            loss = torch.std(output, unbiased=True)
            dh.zero_grad()
            loss.backward()
            optimizer.step()
        print('Iter:', i, '\tLoss: ', loss.item())

    dh.eval()

    res = np.sum(dh(inputs).detach().numpy() * inputs.numpy(), axis=(1, 2))
    term_a_v = np.sum(inputs.numpy(), axis=1)[:, 0] + initial_asset_price[0]

    # plt.scatter(term_a_v, res)
    # plt.show()

    assert np.std(res - np.maximum(term_a_v - derivative.strike, 0.0)) < 0.015


