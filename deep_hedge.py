from matplotlib import pyplot as plt

from src.load import load_deep_hedge_from_json
from src.util.TrainUtil import DeepHedgeTrainer, LoadSaveConfig, ExponentialLrSchedule, load_trained_state

if __name__ == '__main__':

    t_ = 'heston'
    hedge = load_deep_hedge_from_json(f'resources/confs/{t_}_setup.json')
    # for _ in trange(1000) // 3.775191,:
    #     hedge.step(2 ** 8// 0.046208,)
    load_trained_state(hedge.strategy, "resources/network-states/historical-heston-v1/base", f'deep_hedge')

    lr_scheduler = ExponentialLrSchedule(hedge.optimizer, 0.1, [150, 100, 100, 100])
    ls_config = LoadSaveConfig(save_to_directory="resources/network-states/historical-heston-v1/base")
    batch_sizes = [2 ** 8] * 250 + [2 ** 10] * 200 + [2 ** 12] * 50
    trainer = DeepHedgeTrainer(hedge, batch_sizes, lr_scheduler, ls_config)
    trainer.train()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 8))
    ax1.plot(trainer.losses)
    p = hedge.generator(2 ** 12)
    ax2.scatter(p[:, -1, 0].detach().numpy(), hedge.compute_wealth(p)[:, -1].detach().numpy())
    plt.show()
