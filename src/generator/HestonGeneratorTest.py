import numpy as np
import torch
from matplotlib import pyplot as plt

from src.generator.HestonCoefficient import HestonDriftCoefficient, HestonDiffusionCoefficient, \
    HestonCoefficientConfig
from src.util.processes.HestonGenerator import HestonParameterSet, HestonGenerator
from src.generator.EulerGenerator import EulerGenerator, EulerGeneratorConfig
from src.penalty.Augmentations import Scale, Cumsum, AddLags, LeadLag
from src.penalty.SigWassersteinMetric import SignatureConfig, SigWassersteinMetric
from src.util.TimeUtil import UniformTimeDiscretization
from src.util.processes.BlackScholesGenerator import BlackScholesGenerator
from src.util.processes.BrownianMotionGenerator import BrownianMotionGenerator

trading_freq: int = 5
td = UniformTimeDiscretization(trading_freq * 1. / 255., 90 // trading_freq)


def heston_generator_with_sigwasserstein() -> None:

    initializer = HestonParameterSet(
        drift=0.0,
        reversion_level=0.04,
        reversion_speed=2,
        vol_of_vol=0.2,
        correlation=0.8,
    )
    coef_config = HestonCoefficientConfig(initializer, initial_asset_price=1.0)
    drift_coefficient = HestonDriftCoefficient(coef_config)
    diffusion_coefficient = HestonDiffusionCoefficient(coef_config)

    config = EulerGeneratorConfig(td, drift_coefficient.get_initial_asset_price, drift_coefficient, diffusion_coefficient)
    generator = EulerGenerator(generator_config=config)
    optimizer = torch.optim.Adam(generator.parameters())

    bm_increments = torch.as_tensor(
        np.diff(
            BrownianMotionGenerator().generate(
                initial_value=np.zeros((10000, 2)),
                times=td.times,
                random_number_generator=np.random.default_rng(101),
            ), 1, 1,
        ), dtype=torch.float32,
    )

    target_paramters = HestonParameterSet(
        drift=0.0,
        reversion_level=0.04,
        reversion_speed=2,
        vol_of_vol=0.2,
        correlation=0.8,
    )
    heston_process = torch.as_tensor(
        HestonGenerator(parameter=target_paramters).generate(
            initial_value=np.ones((10000, 2))*np.array([1.0, 0.04]),
            times=td.times,
            random_number_generator=np.random.default_rng(404),
        ), dtype=torch.float32
    )

    sig_config = SignatureConfig(depth=2, augmentations=[Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])
    metric = SigWassersteinMetric(heston_process, sig_config)

    for i in range(100):
        output = generator(bm_increments)
        loss = metric(output)
        generator.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            n_params = [f"{n.split('.')[-1]}: {p: .6f}" for n, p in generator.named_parameters()]
            print(f"Iteration: {i}\t Loss: {loss.item(): .6f}\t {'    '.join(n_params)}")

    generator.eval()
    output = generator(bm_increments)
    paths_a = output.detach().numpy()[:100, :, 0].T
    paths_b = output.detach().numpy()[:100, :, 1].T
    paths_c = heston_process.detach().numpy()[:100, :, 0].T
    paths_d = heston_process.detach().numpy()[:100, :, 1].T

    fig, axs = plt.subplots(2, 2, sharey='row', figsize=(6, 6), )
    axs[0, 0].plot(paths_a)
    axs[0, 0].set_title('Out')
    axs[1, 0].plot(paths_b)
    axs[1, 0].set_title('Out')

    axs[0, 1].plot(paths_c)
    axs[0, 1].set_title('In')
    axs[1, 1].plot(paths_d)
    axs[1, 1].set_title('In')
    plt.show()


if __name__ == '__main__':
    heston_generator_with_sigwasserstein()
