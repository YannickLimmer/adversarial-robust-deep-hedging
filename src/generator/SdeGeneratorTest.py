import numpy as np
import torch
from matplotlib import pyplot as plt

from src.generator.EulerGenerator import EulerGeneratorConfig, EulerGenerator
from src.generator.CoefficientNet import CoefficientNetConfig
from src.penalty.Augmentations import Scale, Cumsum, AddLags, LeadLag
from src.penalty.SigWassersteinMetric import SigWassersteinMetric, SignatureConfig
from src.util.TimeUtil import UniformTimeDiscretization
from src.util.processes.BlackScholesGenerator import BlackScholesGenerator
from src.util.processes.BrownianMotionGenerator import BrownianMotionGenerator


def test_simple_generation():

    trading_freq: int = 10
    td = UniformTimeDiscretization(trading_freq * 1. / 255., 90 // trading_freq)
    initial_asset_price = torch.as_tensor([1.0])

    drift_layer_specs = CoefficientNetConfig(initial_asset_price.shape[0], 3, 36)
    sigma_layer_specs = CoefficientNetConfig(initial_asset_price.shape[0], 3, 36, filter=torch.abs)

    gen_specs = EulerGeneratorConfig(
        td=td,
        initial_asset_price=initial_asset_price,
        time_invariant=True,
        drift_specs=drift_layer_specs,
        volatility_specs=sigma_layer_specs,
    )

    generator = EulerGenerator(gen_specs)
    optimizer = torch.optim.Adam(generator.parameters())

    bm_increments = torch.as_tensor(
        np.diff(BrownianMotionGenerator().generate(
            initial_value=np.zeros((10000, initial_asset_price.shape[0])),
            times=td.times,
            random_number_generator=np.random.default_rng(101),
        ), 1, 1), dtype=torch.float32)
    bs_process = torch.as_tensor(BlackScholesGenerator(drift=np.array([0.0]), sigma=np.array([0.2])).generate(
        initial_value=np.ones((10000, 1)),
        times=td.times,
        random_number_generator=np.random.default_rng(104),
    ), dtype=torch.float32)

    for i in range(50):
        output = generator(bm_increments)
        loss = torch.sum(
            torch.abs(torch.std(output[:, :, 0], unbiased=True, dim=0) - torch.std(bs_process, unbiased=True, dim=0)) +
            torch.abs(torch.mean(output[:, :, 0], dim=0) - torch.mean(bs_process, dim=0))
        )
        generator.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            mu = generator.drift(torch.as_tensor([1.0]))
            si = generator.diffusion(torch.as_tensor([1.0]))
            print(f'Iter: {i}\t Loss: {loss.item():.6f}\t Mu: {mu.detach().item():.6f} \t Si: {si.detach().item():.6f}')

    assert loss.item() < 3.85


def run_sigwasserstein_generator() -> None:
    trading_freq: int = 10
    td = UniformTimeDiscretization(trading_freq * 1. / 255., 90 // trading_freq)
    # initial_asset_price = torch.as_tensor([1.0])
    initial_asset_price = torch.as_tensor([1.0, 1.0])

    def zero_drift(x: torch.Tensor) -> torch.Tensor:
        x[0] = 0.0
        return x

    drift_layer_specs = CoefficientNetConfig(initial_asset_price.shape[0], 3, 36, filter=zero_drift)
    sigma_layer_specs = CoefficientNetConfig(initial_asset_price.shape[0], 3, 36, filter=torch.abs)

    gen_specs = EulerGeneratorConfig(td, initial_asset_price, drift_layer_specs, sigma_layer_specs)

    generator = EulerGenerator(gen_specs)
    optimizer = torch.optim.Adam(generator.parameters())

    corr = 0.8
    bm_increments = torch.as_tensor(
        np.diff(
            BrownianMotionGenerator(
                covariance=np.matrix([[1, corr], [corr, 1]]),
            ).generate(
                initial_value=np.zeros((10000, initial_asset_price.shape[0])),
                times=td.times,
                random_number_generator=np.random.default_rng(101),
            ), 1, 1
        ), dtype=torch.float32
    )
    bs_process = torch.as_tensor(
        BlackScholesGenerator(drift=np.array([0.0]), sigma=np.array([0.2])).generate(
            initial_value=np.ones((10000, 1)),
            times=td.times,
            random_number_generator=np.random.default_rng(104),
        ), dtype=torch.float32
    )

    sig_config = SignatureConfig(depth=2, augmentations=[Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])
    metric = SigWassersteinMetric(bs_process, sig_config)

    for i in range(100):
        output = generator(bm_increments)
        loss = metric(output[:, :, 0:1])
        generator.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            mu = generator.drift(torch.as_tensor([td.indices[td.number_of_time_steps // 2], 1.0, 1.0]))
            si = generator.diffusion(torch.as_tensor([td.indices[td.number_of_time_steps // 2], 1.0, 1.0]))
            print(f'Iteration: {i}\t Loss: {loss.item():.6f}\t Mu: {mu.detach().numpy()} \t Si: {si.detach().numpy()}')

    generator.eval()
    output = generator(bm_increments)
    paths_a = output.detach().numpy()[:100, :, 0].T
    paths_b = output.detach().numpy()[:100, :, 1].T
    paths_c = bs_process.detach().numpy()[:100, :, 0].T

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].plot(paths_a)
    axs[1].plot(paths_b)
    axs[2].plot(paths_c)
    plt.show()


if __name__ == '__main__':
    run_sigwasserstein_generator()

